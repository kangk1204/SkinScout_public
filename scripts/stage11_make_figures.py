#!/usr/bin/env python3
"""stage11_make_figures.py — Build publication figure package.

By default this script fails closed when the upstream artifacts needed for a
claim-quality figure package are missing. Use --allow-placeholders only for
explicit draft/scaffold diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import logging
import math
import tempfile
from pathlib import Path

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

LOG = logging.getLogger("stage11.figures")


REQUIRED_NUMERIC_COLUMNS = {"final_score", "score"}

FIGURE_DEFINITIONS = [
    (
        "fig01_workflow.svg",
        "Workflow schematic of the v3 cosmetic-discovery pipeline.",
        ("09_report/index.html",),
    ),
    (
        "fig02_target_landscape.png",
        "Stage 3 top-50 docking score vs PSICHIC, with skin-expression heatmap.",
        ("03_targets/ranked_targets_v3_with_efficacy.csv",),
    ),
    (
        "fig03_pharmacophore.png",
        "Stage 5.5 PLIP/ProLIF pose-supported interaction atom consensus.",
        ("05_pharmacophore/consensus_pharmacophore_atoms.json",),
    ),
    (
        "fig04_analog_scatter.png",
        "Stage 5.6 analog generation — parent→analog Tanimoto vs predicted affinity.",
        ("05_6_analogs/top30_with_properties.csv",),
    ),
    (
        "fig05_md_rmsd.png",
        "Stage 7 MD: RMSD/RMSF + MM-GBSA bar chart per analog.",
        ("07_md/trajectory_index.tsv", "07_md/mmgbsa.tsv"),
    ),
    (
        "fig06_retrosynthesis.png",
        "Stage 7.5 retrosynthesis tree for the final top-3 analogs.",
        ("07_5_retrosynthesis/synthesis_priority_ranking.csv",),
    ),
    (
        "fig07_eval_bars.png",
        "PoseBusters / PLINDER / cold-start / disagreement evaluation summary.",
        ("EVAL:iteration_manifest.json",),
    ),
    (
        "fig08_case_study.png",
        "Recovery of known cosmetic ingredient → known target relationships.",
        ("EVAL:cosmetic_retrospective.csv",),
    ),
]


def expected_outputs(out_dir: Path, out_captions: Path) -> list[Path]:
    return [out_dir / fname for fname, _, _ in FIGURE_DEFINITIONS] + [out_captions]


def remove_outputs(paths: list[Path]) -> None:
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def required_sources(run_dir: Path, eval_dir: Path) -> list[Path]:
    sources: list[Path] = []
    for _, _, rels in FIGURE_DEFINITIONS:
        for rel in rels:
            if rel.startswith("EVAL:"):
                sources.append(eval_dir / rel.split(":", 1)[1])
            else:
                sources.append(run_dir / rel)
    return sources


def _matplotlib_placeholder(path: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        path.write_text("placeholder")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, "data unavailable\n(scaffold placeholder)",
            ha="center", va="center", fontsize=14, alpha=0.6)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _validate_required_numeric_columns(
    path: Path,
    fieldnames: list[str],
    records: list[dict[str, str]],
) -> None:
    for col in sorted(REQUIRED_NUMERIC_COLUMNS & set(fieldnames)):
        for idx, row in enumerate(records):
            value = str(row.get(col, "")).strip()
            if not value:
                raise ValueError(
                    f"column '{col}' must be numeric at row index {idx}: {path}"
                )
            try:
                numeric = float(value)
            except ValueError as exc:
                raise ValueError(
                    f"column '{col}' must be numeric at row index {idx}: {path}"
                ) from exc
            if not math.isfinite(numeric):
                raise ValueError(
                    f"column '{col}' contains non-finite numeric value "
                    f"at row index {idx}: {path}"
                )


def _source_metric(path: Path) -> tuple[str, int]:
    """Return a compact evidence metric for a source artifact."""
    if path.suffix.lower() in {".csv", ".tsv"}:
        try:
            with path.open(newline="") as fh:
                delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
                reader = csv.DictReader(fh, delimiter=delimiter)
                records = list(reader)
            if not records:
                raise ValueError(f"contains no rows: {path}")
            _validate_table_schema(path, reader.fieldnames or [], records)
            _validate_required_numeric_columns(path, reader.fieldnames or [], records)
            id_cols = [
                col
                for col in ("target_id", "id", "known", "target")
                if col in (reader.fieldnames or [])
            ]
            for col in id_cols:
                blank_indexes = [
                    idx for idx, row in enumerate(records)
                    if not str(row.get(col, "")).strip()
                ]
                if blank_indexes:
                    shown = ",".join(str(idx) for idx in blank_indexes[:10])
                    suffix = "..." if len(blank_indexes) > 10 else ""
                    raise ValueError(
                        f"column '{col}' contains blank values at row index(es) "
                        f"{shown}{suffix}: {path}"
                    )
                seen: set[str] = set()
                duplicates: list[str] = []
                for row in records:
                    value = str(row.get(col, "")).strip()
                    if value in seen and value not in duplicates:
                        duplicates.append(value)
                    seen.add(value)
                if duplicates:
                    shown = ",".join(duplicates[:10])
                    suffix = "..." if len(duplicates) > 10 else ""
                    raise ValueError(
                        f"column '{col}' contains duplicate values "
                        f"{shown}{suffix}: {path}"
                    )
            for idx, row in enumerate(records):
                for col, raw in row.items():
                    value = str(raw or "").strip()
                    if not value:
                        continue
                    try:
                        numeric = float(value)
                    except ValueError:
                        continue
                    if not math.isfinite(numeric):
                        raise ValueError(
                            f"column '{col}' contains non-finite numeric value "
                            f"at row index {idx}: {path}"
                        )
            rows = len(records)
            if rows == 0:
                raise ValueError(f"contains no rows: {path}")
            return "rows", rows
        except ValueError:
            raise
        except (csv.Error, UnicodeDecodeError, OSError) as exc:
            raise ValueError(f"failed to parse table source: {path}: {exc}") from exc
    if path.suffix.lower() == ".json":
        try:
            doc = json.loads(path.read_text())
            if path.name == "iteration_manifest.json":
                return "claim-ready eval steps", _validate_iteration_manifest(doc, path)
            if path.name == "consensus_pharmacophore_atoms.json":
                return "confirmed atoms", _validate_pharmacophore_consensus(doc, path)
            if isinstance(doc, list):
                if not doc:
                    raise ValueError(f"contains no items: {path}")
                return "items", len(doc)
            if isinstance(doc, dict):
                if not doc:
                    raise ValueError(f"contains no keys: {path}")
                return "keys", len(doc)
            raise ValueError(f"must be a JSON object or list: {path}")
        except ValueError:
            raise
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
            raise ValueError(f"failed to parse JSON source: {path}: {exc}") from exc
    if path.suffix.lower() in {".html", ".htm"}:
        text = path.read_text(errors="replace")
        lowered = text.lower()
        if "<html" not in lowered or "</html>" not in lowered:
            raise ValueError(f"HTML source is missing document structure: {path}")
        forbidden = ("placeholder", "data unavailable", "scaffold placeholder")
        found = [token for token in forbidden if token in lowered]
        if found:
            raise ValueError(f"HTML source contains placeholder marker(s) {found}: {path}")
        if path.name == "index.html" and ("panel" not in lowered or "target" not in lowered):
            raise ValueError(f"HTML report source lacks expected report panel content: {path}")
        return "bytes", len(text.encode())
    size = path.stat().st_size
    if size == 0:
        raise ValueError(f"contains no bytes: {path}")
    return "bytes", size


def _validate_table_schema(
    path: Path,
    fieldnames: list[str],
    records: list[dict[str, str]],
) -> None:
    if path.name != "ranked_targets_v3_with_efficacy.csv":
        return
    required = {"target_id", "final_score", "source_count", "sources", "efficacy_top1"}
    missing = sorted(required - set(fieldnames))
    if missing:
        raise ValueError(
            f"{path.name} missing required publication source column(s) "
            f"{missing}: {path}"
        )
    for idx, row in enumerate(records):
        source_count_raw = str(row.get("source_count", "")).strip()
        try:
            source_count = int(source_count_raw)
        except ValueError as exc:
            raise ValueError(
                f"{path.name} column 'source_count' must be an integer "
                f"at row index {idx}: {path}"
            ) from exc
        if source_count < 2:
            raise ValueError(
                f"{path.name} column 'source_count' must be >= 2 "
                f"at row index {idx}: {path}"
            )
        sources = str(row.get("sources", "")).strip()
        labels = [part.strip() for part in sources.split(";")]
        if any(label == "" for label in labels):
            raise ValueError(
                f"{path.name} sources contains empty label(s) "
                f"at row index {idx}: {path}"
            )
        if len(labels) != source_count:
            raise ValueError(
                f"{path.name} source_count={source_count} but sources lists "
                f"{len(labels)} label(s) at row index {idx}: {path}"
            )
        duplicate_labels = sorted({
            label for label in labels if labels.count(label) > 1
        })
        if duplicate_labels:
            raise ValueError(
                f"{path.name} sources contains duplicate label(s) "
                f"{duplicate_labels} at row index {idx}: {path}"
            )
        efficacy = str(row.get("efficacy_top1", "")).strip()
        if not efficacy:
            raise ValueError(
                f"{path.name} column 'efficacy_top1' is blank at row index "
                f"{idx}: {path}"
            )


def _source_evidence(path: Path) -> dict[str, object]:
    metric, value = _source_metric(path)
    return {
        "path": str(path),
        "metric": metric,
        "value": value,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _require_int_field(doc: dict[str, object], key: str, path: Path) -> int:
    value = doc.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(
            f"iteration manifest field '{key}' must be a non-negative integer: {path}"
        )
    return value


def _require_status_block(
    doc: dict[str, object],
    key: str,
    expected_status: str,
    path: Path,
) -> dict[str, object]:
    block = doc.get(key)
    if not isinstance(block, dict):
        raise ValueError(f"iteration manifest missing object '{key}': {path}")
    if block.get("status") != expected_status:
        raise ValueError(
            f"iteration manifest {key}.status must be '{expected_status}': {path}"
        )
    return block


def _require_nonblank_text(value: object, label: str, path: Path) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"iteration manifest {label} must be non-empty: {path}")
    return text


def _require_counted_list(
    block: dict[str, object],
    count_key: str,
    list_key: str,
    label: str,
    path: Path,
) -> list[object]:
    count = _require_int_field(block, count_key, path)
    items = block.get(list_key)
    if not isinstance(items, list):
        raise ValueError(
            f"iteration manifest {label}.{list_key} must be a list: {path}"
        )
    if len(items) != count:
        raise ValueError(
            f"iteration manifest {label}.{list_key} length must equal "
            f"{label}.{count_key}: {path}"
        )
    return items


def _require_sha256(value: object, label: str, path: Path) -> None:
    text = str(value or "").strip()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text.lower()):
        raise ValueError(
            f"iteration manifest {label} must be a SHA-256 digest: {path}"
        )


def _require_positive_bytes(value: object, label: str, path: Path) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(
            f"iteration manifest {label} must be a positive byte count: {path}"
        )


def _validate_snapshot_entry(record: dict[str, object], label: str, path: Path) -> None:
    _require_sha256(record.get("sha256"), f"{label}.sha256", path)
    _require_positive_bytes(record.get("bytes"), f"{label}.bytes", path)


def _validate_snapshot_file(record: dict[str, object], label: str, path: Path) -> None:
    _require_nonblank_text(record.get("path"), f"{label}.path", path)
    _safe_relative_path(record.get("relative_path"), f"{label}.relative_path", path)
    _validate_snapshot_entry(record, label, path)


def _validate_snapshot_file_matches(
    record: dict[str, object],
    label: str,
    path: Path,
) -> list[Path]:
    _validate_snapshot_file(record, label, path)
    candidates = _existing_artifact_candidates(
        _require_nonblank_text(record.get("path"), f"{label}.path", path),
        path,
    )
    if not candidates:
        raise ValueError(
            f"iteration manifest {label}.path must exist and be non-empty: {path}"
        )
    expected_bytes = record.get("bytes")
    expected_sha256 = str(record.get("sha256", "")).strip().lower()
    for candidate in candidates:
        if (
            candidate.stat().st_size == expected_bytes
            and hashlib.sha256(candidate.read_bytes()).hexdigest() == expected_sha256
        ):
            return candidates
    raise ValueError(
        f"iteration manifest {label} bytes/sha256 must match existing file: {path}"
    )


def _safe_relative_path(value: object, label: str, path: Path) -> str:
    text = _require_nonblank_text(value, label, path)
    rel_path = Path(text)
    if rel_path.is_absolute() or text == "." or ".." in rel_path.parts:
        raise ValueError(
            f"iteration manifest {label} must be a safe relative path: {path}"
        )
    return str(rel_path)


def _artifact_relative_path_matches(
    candidates: list[Path],
    relative_path: str,
    manifest_path: Path,
) -> bool:
    manifest_dir = manifest_path.parent.resolve()
    for candidate in candidates:
        resolved = candidate.resolve()
        try:
            if str(resolved.relative_to(manifest_dir)) == relative_path:
                return True
        except ValueError:
            pass
    return False


def _sha256_text(value: object, label: str, path: Path) -> str:
    _require_sha256(value, label, path)
    return str(value or "").strip().lower()


def _validate_snapshot_directory_entry_matches(
    record: dict[str, object],
    root: Path,
    label: str,
    path: Path,
) -> Path:
    relative_path = _safe_relative_path(
        record.get("relative_path"),
        f"{label}.relative_path",
        path,
    )
    _validate_snapshot_entry(record, label, path)
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"iteration manifest {label}.relative_path must stay inside "
            f"snapshot directory: {path}"
        ) from exc
    if not candidate.is_file() or candidate.stat().st_size <= 0:
        raise ValueError(
            f"iteration manifest {label} path must exist and be non-empty: {path}"
        )
    if (
        candidate.stat().st_size != record.get("bytes")
        or hashlib.sha256(candidate.read_bytes()).hexdigest()
        != str(record.get("sha256", "")).strip().lower()
    ):
        raise ValueError(
            f"iteration manifest {label} bytes/sha256 must match existing file: {path}"
        )
    return candidate


def _validate_provenance(
    doc: dict[str, object],
    path: Path,
    data_digests: dict[str, str],
) -> None:
    provenance = doc.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"iteration manifest missing object 'provenance': {path}")
    git_commit = _require_nonblank_text(
        provenance.get("git_commit"),
        "provenance.git_commit",
        path,
    )
    if len(git_commit) != 40 or any(
        ch not in "0123456789abcdefABCDEF" for ch in git_commit
    ):
        raise ValueError(
            f"iteration manifest provenance.git_commit must be a 40-hex SHA: {path}"
        )
    config_sha256 = _sha256_text(
        provenance.get("config_sha256"),
        "provenance.config_sha256",
        path,
    )
    workflow_config_sha256 = data_digests.get("workflow_config")
    if config_sha256 != workflow_config_sha256:
        raise ValueError(
            "iteration manifest provenance.config_sha256 must match "
            f"data_snapshot.workflow_config.sha256: {path}"
        )
    tool_versions = provenance.get("tool_versions")
    if not isinstance(tool_versions, dict) or not tool_versions:
        raise ValueError(
            f"iteration manifest provenance.tool_versions must be a non-empty object: {path}"
        )
    for key in ("python", "platform", "rdkit"):
        value = _require_nonblank_text(
            tool_versions.get(key),
            f"provenance.tool_versions.{key}",
            path,
        )
        if key in {"python", "rdkit"} and value.lower() == "missing":
            raise ValueError(
                "iteration manifest provenance.tool_versions must record an "
                f"available {key} version: {path}"
            )
    for key, value in tool_versions.items():
        _require_nonblank_text(key, "provenance.tool_versions key", path)
        _require_nonblank_text(
            value,
            f"provenance.tool_versions.{key}",
            path,
        )


def _validate_manifest_path_bindings(doc: dict[str, object], path: Path) -> None:
    eval_dir_text = _require_nonblank_text(doc.get("eval_dir"), "eval_dir", path)
    eval_dir_candidates = _existing_directory_candidates(eval_dir_text, path)
    if not eval_dir_candidates:
        raise ValueError(
            f"iteration manifest eval_dir must exist and be a directory: {path}"
        )
    if path.parent.resolve() not in eval_dir_candidates:
        raise ValueError(
            f"iteration manifest eval_dir must match eval manifest directory: {path}"
        )

    rankings_dir_text = _require_nonblank_text(
        doc.get("rankings_dir"),
        "rankings_dir",
        path,
    )
    rankings_dir_candidates = _existing_directory_candidates(rankings_dir_text, path)
    expected_rankings_dir = (path.parent / "rankings").resolve()
    if not rankings_dir_candidates:
        raise ValueError(
            f"iteration manifest rankings_dir must exist and be a directory: {path}"
        )
    if expected_rankings_dir not in rankings_dir_candidates:
        raise ValueError(
            "iteration manifest rankings_dir must match eval rankings "
            f"directory: {path}"
        )


def _validate_input_runs(
    doc: dict[str, object],
    path: Path,
    snapshot_paths: set[Path],
) -> None:
    input_runs = _require_status_block(doc, "input_runs", "present", path)
    out_dir_text = _require_nonblank_text(
        input_runs.get("out_dir"),
        "input_runs.out_dir",
        path,
    )
    out_dir_candidates = _existing_directory_candidates(out_dir_text, path)
    if not out_dir_candidates:
        raise ValueError(
            f"iteration manifest input_runs.out_dir must exist and be a directory: {path}"
        )
    if path.parent.resolve() not in out_dir_candidates:
        raise ValueError(
            "iteration manifest input_runs.out_dir must match eval manifest "
            f"directory: {path}"
        )
    rankings_dir_text = _require_nonblank_text(
        input_runs.get("rankings_dir"),
        "input_runs.rankings_dir",
        path,
    )
    rankings_dir_candidates = _existing_directory_candidates(rankings_dir_text, path)
    expected_rankings_dir = (path.parent / "rankings").resolve()
    if not rankings_dir_candidates:
        raise ValueError(
            "iteration manifest input_runs.rankings_dir must exist and be a "
            f"directory: {path}"
        )
    if expected_rankings_dir not in rankings_dir_candidates:
        raise ValueError(
            "iteration manifest input_runs.rankings_dir must match eval "
            f"rankings directory: {path}"
        )
    rankings_roots = set(rankings_dir_candidates)
    n_runs = _require_int_field(input_runs, "n_runs", path)
    if n_runs <= 0:
        raise ValueError(f"iteration manifest input_runs.n_runs must be > 0: {path}")
    runs = input_runs.get("runs")
    if not isinstance(runs, list) or len(runs) != n_runs:
        raise ValueError(
            "iteration manifest input_runs.runs length must equal n_runs: "
            f"{path}"
        )
    seen_run_ids: set[str] = set()
    seen_run_dirs: set[Path] = set()
    for idx, run in enumerate(runs):
        if not isinstance(run, dict):
            raise ValueError(
                "iteration manifest input_runs.runs must contain objects: "
                f"row index {idx}: {path}"
            )
        run_id = _require_nonblank_text(
            run.get("run_id"),
            f"input_runs.runs[{idx}].run_id",
            path,
        )
        if run_id in seen_run_ids:
            raise ValueError(
                f"iteration manifest input_runs contains duplicate run_id "
                f"'{run_id}': {path}"
            )
        seen_run_ids.add(run_id)
        _require_nonblank_text(
            run.get("canonical_smiles"),
            f"input_runs.runs[{idx}].canonical_smiles",
            path,
        )
        run_dir_text = _require_nonblank_text(
            run.get("run_dir"),
            f"input_runs.runs[{idx}].run_dir",
            path,
        )
        run_dir_candidates = _existing_directory_candidates(run_dir_text, path)
        if not run_dir_candidates:
            raise ValueError(
                "iteration manifest input_runs run_dir must exist and be a "
                f"directory: row index {idx}: {path}"
            )
        if any(run_dir in seen_run_dirs for run_dir in run_dir_candidates):
            raise ValueError(
                "iteration manifest input_runs contains duplicate run_dir: "
                f"row index {idx}: {path}"
            )
        seen_run_dirs.update(run_dir_candidates)
        n_leakage_rows = _require_int_field(run, "n_leakage_rows", path)
        if n_leakage_rows <= 0:
            raise ValueError(
                "iteration manifest input_runs.runs n_leakage_rows must be "
                f"> 0: row index {idx}: {path}"
            )
        copied = run.get("copied")
        if not isinstance(copied, list) or not copied:
            raise ValueError(
                "iteration manifest input_runs.runs copied must be a non-empty "
                f"list: row index {idx}: {path}"
            )
        copied_artifacts = run.get("copied_artifacts")
        if (
            not isinstance(copied_artifacts, list)
            or len(copied_artifacts) != len(copied)
        ):
            raise ValueError(
                "iteration manifest input_runs.runs copied_artifacts must be a "
                f"list matching copied length: row index {idx}: {path}"
            )
        seen_copied_paths: set[str] = set()
        seen_copied_artifact_paths: set[Path] = set()
        for copied_idx, copied_value in enumerate(copied):
            copied_path = _require_nonblank_text(
                copied_value,
                f"input_runs.runs[{idx}].copied[{copied_idx}]",
                path,
            )
            if copied_path in seen_copied_paths:
                raise ValueError(
                    "iteration manifest input_runs copied contains duplicate "
                    f"path: row index {idx}, copied index {copied_idx}: {path}"
                )
            seen_copied_paths.add(copied_path)
            copied_candidates = _existing_artifact_candidates(copied_path, path)
            if not copied_candidates:
                raise ValueError(
                    "iteration manifest input_runs copied artifact must exist "
                    f"and be non-empty: row index {idx}, copied index "
                    f"{copied_idx}: {path}"
                )
            if any(
                candidate in seen_copied_artifact_paths
                for candidate in copied_candidates
            ):
                raise ValueError(
                    "iteration manifest input_runs copied contains duplicate "
                    f"path: row index {idx}, copied index {copied_idx}: {path}"
                )
            seen_copied_artifact_paths.update(copied_candidates)
            copied_record = copied_artifacts[copied_idx]
            if not isinstance(copied_record, dict):
                raise ValueError(
                    "iteration manifest input_runs.runs copied_artifacts must "
                    f"contain objects: row index {idx}, copied index "
                    f"{copied_idx}: {path}"
                )
            record_path = _require_nonblank_text(
                copied_record.get("path"),
                f"input_runs.runs[{idx}].copied_artifacts[{copied_idx}].path",
                path,
            )
            if record_path != copied_path:
                raise ValueError(
                    "iteration manifest input_runs copied_artifacts path must "
                    f"match copied path: row index {idx}, copied index "
                    f"{copied_idx}: {path}"
                )
            source_path = _require_nonblank_text(
                copied_record.get("source_path"),
                f"input_runs.runs[{idx}].copied_artifacts[{copied_idx}].source_path",
                path,
            )
            source_candidates = _existing_artifact_candidates(source_path, path)
            if not source_candidates:
                raise ValueError(
                    "iteration manifest input_runs copied_artifacts source_path "
                    f"must exist and be non-empty: row index {idx}, copied "
                    f"index {copied_idx}: {path}"
                )
            if not any(
                _is_relative_to(source_candidate, run_dir_candidate)
                for source_candidate in source_candidates
                for run_dir_candidate in run_dir_candidates
            ):
                raise ValueError(
                    "iteration manifest input_runs copied_artifacts source_path "
                    f"must be inside run_dir: row index {idx}, copied index "
                    f"{copied_idx}: {path}"
                )
            _require_positive_bytes(
                copied_record.get("source_bytes"),
                (
                    f"input_runs.runs[{idx}].copied_artifacts"
                    f"[{copied_idx}].source_bytes"
                ),
                path,
            )
            _require_sha256(
                copied_record.get("source_sha256"),
                (
                    f"input_runs.runs[{idx}].copied_artifacts"
                    f"[{copied_idx}].source_sha256"
                ),
                path,
            )
            expected_source_bytes = copied_record.get("source_bytes")
            expected_source_sha256 = (
                str(copied_record.get("source_sha256", "")).strip().lower()
            )
            if not any(
                candidate.stat().st_size == expected_source_bytes
                and hashlib.sha256(candidate.read_bytes()).hexdigest()
                == expected_source_sha256
                for candidate in source_candidates
            ):
                raise ValueError(
                    "iteration manifest input_runs copied_artifacts "
                    f"source_bytes/source_sha256 must match existing source: "
                    f"row index {idx}, copied index {copied_idx}: {path}"
                )
            _validate_snapshot_entry(
                copied_record,
                f"input_runs.runs[{idx}].copied_artifacts[{copied_idx}]",
                path,
            )
            if not any(
                _is_relative_to(candidate, rankings_root)
                for candidate in copied_candidates
                for rankings_root in rankings_roots
            ):
                raise ValueError(
                    "iteration manifest input_runs copied artifact must be "
                    f"inside rankings_dir: row index {idx}, copied index "
                    f"{copied_idx}: {path}"
                )
            expected_bytes = copied_record.get("bytes")
            expected_sha256 = str(copied_record.get("sha256", "")).strip().lower()
            if not any(
                candidate.stat().st_size == expected_bytes
                and hashlib.sha256(candidate.read_bytes()).hexdigest() == expected_sha256
                for candidate in copied_candidates
            ):
                raise ValueError(
                    "iteration manifest input_runs copied_artifacts "
                    f"bytes/sha256 must match existing file: row index {idx}, "
                    f"copied index {copied_idx}: {path}"
                )
            if not any(candidate in snapshot_paths for candidate in copied_candidates):
                raise ValueError(
                    "iteration manifest input_runs copied artifact must be "
                    f"included in artifact_snapshot: row index {idx}, copied "
                    f"index {copied_idx}: {path}"
                )


def _validate_input_status(doc: dict[str, object], path: Path) -> None:
    input_status = _require_status_block(doc, "input_status", "ok", path)
    checked = _require_counted_list(
        input_status,
        "n_checked",
        "checked",
        "input_status",
        path,
    )
    incomplete = _require_counted_list(
        input_status,
        "n_incomplete",
        "incomplete",
        "input_status",
        path,
    )
    if not checked:
        raise ValueError(f"iteration manifest has no checked eval input sets: {path}")
    if incomplete:
        raise ValueError(f"iteration manifest has incomplete eval inputs: {path}")
    seen_names: set[str] = set()
    seen_present_paths: set[Path] = set()
    for idx, record in enumerate(checked):
        if not isinstance(record, dict):
            raise ValueError(
                "iteration manifest input_status.checked must contain objects: "
                f"row index {idx}: {path}"
            )
        name = _require_nonblank_text(
            record.get("name"),
            f"input_status.checked[{idx}].name",
            path,
        )
        if name in seen_names:
            raise ValueError(
                "iteration manifest input_status.checked contains duplicate "
                f"name: {name}: row index {idx}: {path}"
            )
        seen_names.add(name)
        present = record.get("present")
        if not isinstance(present, list) or not present:
            raise ValueError(
                "iteration manifest input_status.checked present lists must be "
                f"non-empty: row index {idx}: {path}"
            )
        for present_idx, present_path_obj in enumerate(present):
            present_path = _require_nonblank_text(
                present_path_obj,
                (
                    f"input_status.checked[{idx}].present"
                    f"[{present_idx}]"
                ),
                path,
            )
            present_candidates = _existing_artifact_candidates(present_path, path)
            if not present_candidates:
                raise ValueError(
                    "iteration manifest input_status.checked present path "
                    "must exist and be non-empty: "
                    f"row index {idx}, present index {present_idx}: {path}"
                )
            if any(candidate in seen_present_paths for candidate in present_candidates):
                raise ValueError(
                    "iteration manifest input_status.checked present contains "
                    "duplicate path: "
                    f"row index {idx}, present index {present_idx}: {path}"
                )
            seen_present_paths.update(present_candidates)
        missing = record.get("missing")
        if not isinstance(missing, list) or missing:
            raise ValueError(
                "iteration manifest input_status.checked missing lists must be "
                f"empty: row index {idx}: {path}"
            )


def _validate_leakage_status(doc: dict[str, object], path: Path) -> None:
    leakage_status = _require_status_block(doc, "leakage_status", "ok", path)
    n_rows = _require_int_field(leakage_status, "n_rows", path)
    if n_rows <= 0:
        raise ValueError(
            f"iteration manifest leakage_status.n_rows must be > 0: {path}"
        )
    for key in ("n_incomplete", "n_leak_flags", "n_invalid_rows"):
        if _require_int_field(leakage_status, key, path) != 0:
            raise ValueError(
                f"iteration manifest leakage_status.{key} must be 0: {path}"
            )


def _existing_artifact_candidates(text: str, manifest_path: Path) -> list[Path]:
    raw_path = Path(text)
    candidates = (
        [raw_path]
        if raw_path.is_absolute()
        else [Path.cwd() / raw_path, manifest_path.parent / raw_path]
    )
    return [
        candidate.resolve()
        for candidate in candidates
        if candidate.is_file() and candidate.stat().st_size > 0
    ]


def _existing_directory_candidates(text: str, manifest_path: Path) -> list[Path]:
    raw_path = Path(text)
    candidates = (
        [raw_path]
        if raw_path.is_absolute()
        else [Path.cwd() / raw_path, manifest_path.parent / raw_path]
    )
    return [candidate.resolve() for candidate in candidates if candidate.is_dir()]


def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _validate_threshold_status(
    doc: dict[str, object],
    path: Path,
    step_outputs: set[Path],
) -> list[str]:
    threshold_status = _require_status_block(doc, "threshold_status", "ok", path)
    checked = _require_counted_list(
        threshold_status,
        "n_checked",
        "checked",
        "threshold_status",
        path,
    )
    failures = _require_counted_list(
        threshold_status,
        "n_failed",
        "failures",
        "threshold_status",
        path,
    )
    missing_required = _require_counted_list(
        threshold_status,
        "n_missing_required",
        "missing_required",
        "threshold_status",
        path,
    )
    if not checked:
        raise ValueError(f"iteration manifest has no checked thresholds: {path}")
    if failures:
        raise ValueError(f"iteration manifest records threshold failures: {path}")
    if missing_required:
        raise ValueError(
            f"iteration manifest records threshold metrics missing required "
            f"passes_threshold columns: {path}"
        )
    metric_names: list[str] = []
    seen_threshold_paths: set[Path] = set()
    seen_metric_names: set[str] = set()
    for idx, record in enumerate(checked):
        if not isinstance(record, dict):
            raise ValueError(
                "iteration manifest threshold_status.checked must contain "
                f"objects: row index {idx}: {path}"
            )
        threshold_path = _require_nonblank_text(
            record.get("path"),
            f"threshold_status.checked[{idx}].path",
            path,
        )
        threshold_candidates = _existing_artifact_candidates(threshold_path, path)
        if not threshold_candidates:
            raise ValueError(
                "iteration manifest threshold_status.checked path must exist "
                f"and be non-empty: row index {idx}: {path}"
            )
        canonical_threshold_path = threshold_candidates[0]
        if canonical_threshold_path in seen_threshold_paths:
            raise ValueError(
                "iteration manifest threshold_status.checked contains "
                f"duplicate path: row index {idx}: {path}"
            )
        seen_threshold_paths.add(canonical_threshold_path)
        if not any(candidate in step_outputs for candidate in threshold_candidates):
            raise ValueError(
                "iteration manifest threshold_status.checked path must match "
                f"a passed eval step output: row index {idx}: {path}"
            )
        metric_name = Path(threshold_path).stem
        if metric_name in seen_metric_names:
            raise ValueError(
                "iteration manifest threshold_status.checked contains "
                f"duplicate metric name '{metric_name}': row index {idx}: {path}"
            )
        seen_metric_names.add(metric_name)
        metric_names.append(metric_name)
        if record.get("passed") is not True:
            raise ValueError(
                "iteration manifest threshold_status.checked rows must be "
                f"passed: row index {idx}: {path}"
            )
        for key in ("n_rows", "n_failed_rows", "n_invalid_rows"):
            _require_int_field(record, key, path)
        if record.get("n_rows") <= 0:
            raise ValueError(
                "iteration manifest threshold_status.checked n_rows must be "
                f"> 0: row index {idx}: {path}"
            )
        if record.get("n_failed_rows") != 0 or record.get("n_invalid_rows") != 0:
            raise ValueError(
                "iteration manifest threshold_status.checked rows must have "
                f"zero failed/invalid rows: row index {idx}: {path}"
            )
    return metric_names


def _validate_ranking_metrics(
    doc: dict[str, object],
    threshold_metrics: list[str],
    path: Path,
) -> None:
    metrics = doc.get("ranking_metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError(f"iteration manifest missing object 'ranking_metrics': {path}")
    threshold_metric_names = set(threshold_metrics)
    ranking_metric_names = [
        _require_nonblank_text(metric_name, "ranking_metrics key", path)
        for metric_name in metrics
    ]
    unsupported_metrics = sorted(
        metric_name
        for metric_name in ranking_metric_names
        if metric_name not in threshold_metric_names
    )
    if unsupported_metrics:
        raise ValueError(
            "iteration manifest ranking_metrics contains metrics without "
            f"threshold evidence {unsupported_metrics[:5]}: {path}"
        )
    for metric_name in threshold_metrics:
        record = metrics.get(metric_name)
        if not isinstance(record, dict):
            raise ValueError(
                "iteration manifest ranking_metrics missing thresholded metric "
                f"'{metric_name}': {path}"
            )
        if record.get("passes_threshold") is not True:
            raise ValueError(
                "iteration manifest ranking_metrics threshold metric must pass: "
                f"{metric_name}: {path}"
            )
        if record.get("passes_threshold_valid") is not True:
            raise ValueError(
                "iteration manifest ranking_metrics threshold validity must be "
                f"true: {metric_name}: {path}"
            )
        if metric_name in {"cosmetic_retrospective", "skin_efficacy_recovery"}:
            n_evaluated = _require_int_field(record, "n_evaluated", path)
            n_no_ranking = _require_int_field(record, "n_no_ranking", path)
            if n_evaluated <= 0:
                raise ValueError(
                    "iteration manifest ranking_metrics direct evaluator must "
                    f"record evaluated rows: {metric_name}: {path}"
                )
            if n_no_ranking != 0:
                raise ValueError(
                    "iteration manifest ranking_metrics direct evaluator "
                    f"contains no_ranking diagnostics: {metric_name}: {path}"
                )


def _validate_eval_steps(steps: list[object], path: Path) -> set[Path]:
    seen_names: set[str] = set()
    seen_output_paths: set[Path] = set()
    output_paths: set[Path] = set()
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(
                "iteration manifest steps must contain objects: "
                f"row index {idx}: {path}"
            )
        name = _require_nonblank_text(step.get("name"), f"steps[{idx}].name", path)
        if name in seen_names:
            raise ValueError(
                f"iteration manifest steps contains duplicate name '{name}': {path}"
            )
        seen_names.add(name)
        if step.get("status") != "passed":
            raise ValueError(
                f"iteration manifest step '{name}' status must be 'passed': {path}"
            )
        for key in ("command", "outputs"):
            values = step.get(key)
            if not isinstance(values, list) or not values:
                raise ValueError(
                    f"iteration manifest steps[{idx}].{key} must be a "
                    f"non-empty list: {path}"
                )
            for value_idx, value in enumerate(values):
                text = _require_nonblank_text(
                    value,
                    f"steps[{idx}].{key}[{value_idx}]",
                    path,
                )
                if key == "outputs":
                    candidates = _existing_artifact_candidates(text, path)
                    if not candidates:
                        raise ValueError(
                            "iteration manifest step output must exist and be "
                            f"non-empty: steps[{idx}].outputs[{value_idx}]: "
                            f"{text}: {path}"
                        )
                    duplicate_outputs = [
                        candidate
                        for candidate in candidates
                        if candidate in seen_output_paths
                    ]
                    if duplicate_outputs:
                        raise ValueError(
                            "iteration manifest steps contain duplicate output "
                            f"path: steps[{idx}].outputs[{value_idx}]: "
                            f"{text}: {path}"
                        )
                    seen_output_paths.update(candidates)
                    output_paths.update(candidates)
    return output_paths


def _validate_data_snapshot(doc: dict[str, object], path: Path) -> dict[str, str]:
    data_snapshot = doc.get("data_snapshot")
    if not isinstance(data_snapshot, dict):
        raise ValueError(f"iteration manifest missing object 'data_snapshot': {path}")
    required = (
        "workflow_config",
        "target_classes",
        "chembl_fingerprints",
        "activity_retrieval_gate",
        "training_sequence_db",
        "training_ligands",
        "training_holo",
    )
    missing = [key for key in required if key not in data_snapshot]
    if missing:
        raise ValueError(
            f"iteration manifest data_snapshot missing required entries "
            f"{missing}: {path}"
        )
    digests: dict[str, str] = {}
    seen_file_paths: set[Path] = set()
    for key in required:
        record = data_snapshot[key]
        if not isinstance(record, dict):
            raise ValueError(
                f"iteration manifest data_snapshot.{key} must be an object: {path}"
            )
        if record.get("status") != "present":
            raise ValueError(
                f"iteration manifest data_snapshot.{key}.status must be "
                f"'present': {path}"
            )
        kind = record.get("kind")
        if kind == "directory":
            root = Path(
                _require_nonblank_text(
                    record.get("path"),
                    f"data_snapshot.{key}.path",
                    path,
                )
            )
            if not root.is_absolute():
                root = (path.parent / root).resolve()
            if not root.is_dir():
                raise ValueError(
                    f"iteration manifest data_snapshot.{key}.path must be an "
                    f"existing directory: {path}"
                )
            n_files = record.get("n_files")
            if not isinstance(n_files, int) or isinstance(n_files, bool) or n_files <= 0:
                raise ValueError(
                    f"iteration manifest data_snapshot.{key}.n_files must be "
                    f"positive: {path}"
                )
            entries = record.get("entries")
            if not isinstance(entries, list) or len(entries) != n_files:
                raise ValueError(
                    f"iteration manifest data_snapshot.{key}.entries length "
                    f"must equal n_files: {path}"
                )
            seen_relative_paths: set[str] = set()
            for idx, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    raise ValueError(
                        f"iteration manifest data_snapshot.{key}.entries "
                        f"must contain objects: {path}"
                    )
                candidate = _validate_snapshot_directory_entry_matches(
                    entry,
                    root,
                    f"data_snapshot.{key}.entries[{idx}]",
                    path,
                )
                relative_path = str(entry.get("relative_path", "")).strip()
                if relative_path in seen_relative_paths:
                    raise ValueError(
                        "iteration manifest data_snapshot directory entries "
                        f"contain duplicate relative_path '{relative_path}': "
                        f"{key}: {path}"
                    )
                seen_relative_paths.add(relative_path)
                if candidate in seen_file_paths:
                    raise ValueError(
                        "iteration manifest data_snapshot contains duplicate "
                        f"file path: {key}: {path}"
                    )
                seen_file_paths.add(candidate)
        elif kind == "file":
            candidates = _validate_snapshot_file_matches(
                record,
                f"data_snapshot.{key}",
                path,
            )
            duplicate_paths = [
                candidate
                for candidate in candidates
                if candidate in seen_file_paths
            ]
            if duplicate_paths:
                raise ValueError(
                    "iteration manifest data_snapshot contains duplicate "
                    f"file path: {key}: {path}"
                )
            seen_file_paths.update(candidates)
            digests[key] = str(record.get("sha256", "")).strip().lower()
        else:
            raise ValueError(
                f"iteration manifest data_snapshot.{key}.kind must be "
                f"'file' or 'directory': {path}"
            )
    return digests


def _validate_activity_retrieval_gate(
    doc: dict[str, object],
    manifest_path: Path,
    data_digests: dict[str, str],
) -> None:
    record = doc.get("activity_retrieval_gate")
    if not isinstance(record, dict):
        raise ValueError(
            f"iteration manifest activity_retrieval_gate must be an object: {manifest_path}"
        )
    if (
        record.get("status") != "pass"
        or record.get("schema_version") != ACTIVITY_RETRIEVAL_GATE_SCHEMA
    ):
        raise ValueError(
            f"iteration manifest activity_retrieval_gate must be claim-ready: {manifest_path}"
        )
    path_text = _require_nonblank_text(
        record.get("path"),
        "activity_retrieval_gate.path",
        manifest_path,
    )
    gate_path = Path(path_text)
    if not gate_path.is_absolute():
        gate_path = (manifest_path.parent / gate_path).resolve()
    if gate_path.is_symlink() or not gate_path.is_file() or gate_path.stat().st_size <= 0:
        raise ValueError(
            f"iteration manifest activity retrieval gate is missing or unsafe: {gate_path}"
        )
    digest = hashlib.sha256(gate_path.read_bytes()).hexdigest()
    if (
        record.get("bytes") != gate_path.stat().st_size
        or record.get("sha256") != digest
        or data_digests.get("activity_retrieval_gate") != digest
    ):
        raise ValueError(
            f"iteration manifest activity retrieval gate binding mismatch: {manifest_path}"
        )
    try:
        check_activity_retrieval_gate(gate_path)
    except (OSError, ValueError, SystemExit) as exc:
        raise ValueError(
            f"iteration manifest activity retrieval gate failed revalidation: {exc}"
        ) from exc


def _validate_artifact_snapshot(
    doc: dict[str, object],
    path: Path,
    step_outputs: set[Path],
) -> set[Path]:
    artifact_snapshot = doc.get("artifact_snapshot")
    if not isinstance(artifact_snapshot, list) or not artifact_snapshot:
        raise ValueError(
            f"iteration manifest artifact_snapshot must be a non-empty list: {path}"
        )
    snapshot_paths: set[Path] = set()
    seen_snapshot_paths: set[Path] = set()
    seen_relative_paths: set[str] = set()
    for idx, record in enumerate(artifact_snapshot):
        if not isinstance(record, dict):
            raise ValueError(
                "iteration manifest artifact_snapshot must contain objects: "
                f"row index {idx}: {path}"
            )
        candidates = _validate_snapshot_file_matches(
            record,
            f"artifact_snapshot[{idx}]",
            path,
        )
        relative_path = _safe_relative_path(
            record.get("relative_path"),
            f"artifact_snapshot[{idx}].relative_path",
            path,
        )
        duplicate_paths = [
            candidate
            for candidate in candidates
            if candidate in seen_snapshot_paths
        ]
        if duplicate_paths:
            raise ValueError(
                "iteration manifest artifact_snapshot contains duplicate path: "
                f"row index {idx}: {path}"
            )
        if not _artifact_relative_path_matches(candidates, relative_path, path):
            raise ValueError(
                "iteration manifest artifact_snapshot relative_path must match "
                f"artifact path: row index {idx}: {path}"
            )
        if relative_path in seen_relative_paths:
            raise ValueError(
                "iteration manifest artifact_snapshot contains duplicate "
                f"relative_path '{relative_path}': {path}"
            )
        seen_relative_paths.add(relative_path)
        seen_snapshot_paths.update(candidates)
        snapshot_paths.update(candidates)
    missing_step_outputs = sorted(
        str(step_output)
        for step_output in step_outputs
        if step_output not in snapshot_paths
    )
    if missing_step_outputs:
        shown = ", ".join(missing_step_outputs[:5])
        raise ValueError(
            "iteration manifest artifact_snapshot must include every passed "
            f"eval step output: {shown}: {path}"
        )
    return snapshot_paths


def _min_source_count_for_ranking(ranking: str) -> int | None:
    ranking_path = Path(ranking)
    if ranking_path.name == "cold_start__dti_only.csv":
        return None
    if ranking_path.name == "cold_start__fast.csv":
        return 2
    if ranking_path.name == "cold_start__comprehensive.csv":
        return 3
    if ranking_path.parent.name == "cosmetic_retro":
        return 2
    return None


def _source_labels(value: object, label: str, path: Path) -> list[str]:
    text = _require_nonblank_text(value, label, path)
    labels = [part.strip() for part in text.split(";")]
    if any(label == "" for label in labels):
        raise ValueError(
            f"iteration manifest {label} contains empty source labels: {path}"
        )
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        shown = ", ".join(duplicates[:10])
        raise ValueError(
            f"iteration manifest {label} contains duplicate source labels "
            f"{shown}: {path}"
        )
    return labels


def _positive_int_text(value: object, label: str, path: Path) -> int:
    text = _require_nonblank_text(value, label, path)
    try:
        number = int(text)
    except ValueError as exc:
        raise ValueError(
            f"iteration manifest {label} must be a positive integer: {path}"
        ) from exc
    if str(number) != text or number <= 0:
        raise ValueError(
            f"iteration manifest {label} must be a positive integer: {path}"
        )
    return number


RANKING_SCORE_COLUMNS = (
    "final_score",
    "rrf_score",
    "score",
    "skin_weighted_score",
    "dock_score",
    "psichic_score",
)


def _ranking_score_column(row: dict[str, str], ranking: str, path: Path) -> str:
    for col in RANKING_SCORE_COLUMNS:
        if col in row and str(row.get(col, "")).strip():
            return col
    raise ValueError(
        "iteration manifest top_target_rationale ranking top row must include "
        f"a score column: {ranking}: {path}"
    )


def _ranking_score_value(
    row: dict[str, str],
    col: str,
    ranking: str,
    path: Path,
    idx: int,
) -> float:
    raw = str(row.get(col, "")).strip()
    if not raw:
        raise ValueError(
            "iteration manifest top_target_rationale ranking score column "
            f"'{col}' must be non-empty: {ranking}: row index {idx}: {path}"
        )
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            "iteration manifest top_target_rationale ranking score column "
            f"'{col}' must be numeric: {ranking}: row index {idx}: {path}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            "iteration manifest top_target_rationale ranking score column "
            f"'{col}' must be finite: {ranking}: row index {idx}: {path}"
        )
    return value


def _validate_ranking_top_score(
    records: list[dict[str, str]],
    col: str,
    ranking: str,
    path: Path,
) -> None:
    scores = [
        _ranking_score_value(row, col, ranking, path, idx)
        for idx, row in enumerate(records)
    ]
    top_score = scores[0]
    best_score = max(scores)
    if best_score > top_score and not math.isclose(
        best_score,
        top_score,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "iteration manifest top_target_rationale ranking top row must "
            f"have the best {col}: {ranking}: {path}"
        )
    tied_top_indexes = [
        idx
        for idx, score in enumerate(scores)
        if math.isclose(score, top_score, rel_tol=1e-9, abs_tol=1e-12)
    ]
    if len(tied_top_indexes) > 1:
        shown = ",".join(str(idx) for idx in tied_top_indexes[:10])
        suffix = "..." if len(tied_top_indexes) > 10 else ""
        raise ValueError(
            "iteration manifest top_target_rationale ranking top row "
            f"{col} is tied at row index(es) {shown}{suffix}: {ranking}: {path}"
        )


def _validate_ranking_sources(
    records: list[dict[str, str]],
    ranking: str,
    path: Path,
) -> None:
    min_source_count = _min_source_count_for_ranking(ranking)
    if min_source_count is None:
        return
    for idx, row in enumerate(records):
        source_count = _positive_int_text(
            row.get("source_count"),
            f"top_target_rationale ranking source_count for {ranking} row index {idx}",
            path,
        )
        labels = _source_labels(
            row.get("sources"),
            f"top_target_rationale ranking sources for {ranking} row index {idx}",
            path,
        )
        if source_count != len(labels):
            raise ValueError(
                "iteration manifest top_target_rationale ranking "
                f"source_count={source_count} but sources lists {len(labels)} "
                f"label(s): {ranking}: row index {idx}: {path}"
            )
        if source_count < min_source_count:
            raise ValueError(
                "iteration manifest top_target_rationale ranking source_count "
                f"must be >= {min_source_count}: {ranking}: row index {idx}: {path}"
            )


def _validate_ranking_skin_scores(
    records: list[dict[str, str]],
    ranking: str,
    path: Path,
) -> None:
    if "skin_score" not in records[0]:
        return
    for idx, row in enumerate(records):
        value = str(row.get("skin_score", "")).strip()
        if not value:
            raise ValueError(
                "iteration manifest top_target_rationale ranking skin_score "
                f"must be non-empty: {ranking}: row index {idx}: {path}"
            )
        _fraction_value(
            value,
            f"top_target_rationale ranking skin_score for {ranking} row index {idx}",
            path,
        )


def _validate_ranking_text_rationale(
    records: list[dict[str, str]],
    ranking: str,
    path: Path,
) -> None:
    for col in ("efficacy_category", "cosmetic_decision", "drug_decision"):
        if col not in records[0]:
            continue
        for idx, row in enumerate(records):
            value = row.get(col, "")
            if value is None or not str(value).strip():
                raise ValueError(
                    "iteration manifest top_target_rationale ranking column "
                    f"'{col}' is blank: {ranking}: row index {idx}: {path}"
                )


def _ranking_top_row(
    ranking: str,
    manifest_path: Path,
) -> tuple[Path, dict[str, str]]:
    candidates = _existing_artifact_candidates(ranking, manifest_path)
    if not candidates:
        raise ValueError(
            "iteration manifest top_target_rationale ranking must exist and "
            f"be non-empty: {ranking}: {manifest_path}"
        )
    ranking_path = candidates[0]
    try:
        with ranking_path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            records = list(reader)
    except (csv.Error, UnicodeDecodeError, OSError) as exc:
        raise ValueError(
            "iteration manifest top_target_rationale ranking failed to parse: "
            f"{ranking}: {manifest_path}: {exc}"
        ) from exc
    if not records:
        raise ValueError(
            "iteration manifest top_target_rationale ranking contains no rows: "
            f"{ranking}: {manifest_path}"
        )
    seen_target_ids: set[str] = set()
    for idx, row in enumerate(records):
        extra_fields = row.get(None)
        if extra_fields:
            raise ValueError(
                "iteration manifest top_target_rationale ranking row has "
                f"extra field(s): {ranking}: row index {idx}: {manifest_path}"
            )
        target_id = str(row.get("target_id", "")).strip()
        if not target_id:
            raise ValueError(
                "iteration manifest top_target_rationale ranking target_id "
                f"must be non-empty: {ranking}: row index {idx}: {manifest_path}"
            )
        if target_id in seen_target_ids:
            raise ValueError(
                "iteration manifest top_target_rationale ranking contains "
                f"duplicate target_id '{target_id}': {ranking}: {manifest_path}"
            )
        seen_target_ids.add(target_id)
    score_col = _ranking_score_column(records[0], ranking, manifest_path)
    _validate_ranking_top_score(records, score_col, ranking, manifest_path)
    _validate_ranking_sources(records, ranking, manifest_path)
    _validate_ranking_skin_scores(records, ranking, manifest_path)
    _validate_ranking_text_rationale(records, ranking, manifest_path)
    return ranking_path, records[0]


def _top_row_score(row: dict[str, str], ranking: str, path: Path) -> float:
    score_col = _ranking_score_column(row, ranking, path)
    return _ranking_score_value(row, score_col, ranking, path, 0)


def _fraction_value(value: object, label: str, path: Path) -> float:
    if isinstance(value, bool):
        raise ValueError(f"iteration manifest {label} must be in [0, 1]: {path}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"iteration manifest {label} must be numeric: {path}"
        ) from exc
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise ValueError(f"iteration manifest {label} must be in [0, 1]: {path}")
    return number


def _validate_top_target_rationale(
    doc: dict[str, object],
    path: Path,
    snapshot_paths: set[Path],
) -> None:
    rationale_rows = doc.get("top_target_rationale")
    if not isinstance(rationale_rows, list) or not rationale_rows:
        raise ValueError(
            "iteration manifest top_target_rationale must be a non-empty list: "
            f"{path}"
        )
    seen_ranking_paths: set[Path] = set()
    for idx, row in enumerate(rationale_rows):
        if not isinstance(row, dict):
            raise ValueError(
                "iteration manifest top_target_rationale must contain objects: "
                f"row index {idx}: {path}"
            )
        ranking = _require_nonblank_text(
            row.get("ranking"),
            f"top_target_rationale[{idx}].ranking",
            path,
        )
        ranking_path, top_row = _ranking_top_row(ranking, path)
        if ranking_path in seen_ranking_paths:
            raise ValueError(
                "iteration manifest top_target_rationale contains duplicate "
                f"ranking: row index {idx}: {path}"
            )
        seen_ranking_paths.add(ranking_path)
        if ranking_path not in snapshot_paths:
            raise ValueError(
                "iteration manifest top_target_rationale ranking must be "
                f"included in artifact_snapshot: row index {idx}: {path}"
            )
        target_id = _require_nonblank_text(
            row.get("target_id"),
            f"top_target_rationale[{idx}].target_id",
            path,
        )
        top_target_id = str(top_row.get("target_id", "")).strip()
        if top_target_id != target_id:
            raise ValueError(
                "iteration manifest top_target_rationale target_id must match "
                f"ranking top row: row index {idx}: {path}"
            )
        score = row.get("score")
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(float(score))
        ):
            raise ValueError(
                "iteration manifest top_target_rationale score must be finite: "
                f"row index {idx}: {path}"
            )
        top_score = _top_row_score(top_row, ranking, path)
        if not math.isclose(float(score), top_score, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(
                "iteration manifest top_target_rationale score must match "
                f"ranking top row: row index {idx}: {path}"
            )
        rationale = row.get("rationale")
        if not isinstance(rationale, dict):
            raise ValueError(
                "iteration manifest top_target_rationale.rationale must be an "
                f"object: row index {idx}: {path}"
            )
        for key, value in rationale.items():
            key_text = _require_nonblank_text(
                key,
                f"top_target_rationale[{idx}].rationale key",
                path,
            )
            if isinstance(value, str):
                _require_nonblank_text(
                    value,
                    f"top_target_rationale[{idx}].rationale.{key_text}",
                    path,
                )
        source_count = rationale.get("source_count")
        if (
            not isinstance(source_count, int)
            or isinstance(source_count, bool)
            or source_count <= 0
        ):
            raise ValueError(
                "iteration manifest top_target_rationale rationale.source_count "
                f"must be a positive integer: row index {idx}: {path}"
            )
        labels = _source_labels(
            rationale.get("sources"),
            f"top_target_rationale[{idx}].rationale.sources",
            path,
        )
        if source_count != len(labels):
            raise ValueError(
                "iteration manifest top_target_rationale rationale.source_count "
                f"{source_count} does not match sources list length "
                f"{len(labels)}: row index {idx}: {path}"
            )
        min_source_count = _min_source_count_for_ranking(ranking)
        if min_source_count is not None and source_count < min_source_count:
            raise ValueError(
                "iteration manifest top_target_rationale rationale.source_count "
                f"must be >= {min_source_count} for ranking '{ranking}': "
                f"row index {idx}: {path}"
            )
        top_source_count = str(top_row.get("source_count", "")).strip()
        top_sources = str(top_row.get("sources", "")).strip()
        if str(source_count) != top_source_count:
            raise ValueError(
                "iteration manifest top_target_rationale rationale.source_count "
                f"must match ranking top row: row index {idx}: {path}"
            )
        if ";".join(labels) != top_sources:
            raise ValueError(
                "iteration manifest top_target_rationale rationale.sources "
                f"must match ranking top row: row index {idx}: {path}"
            )
        if "skin_score" in rationale:
            skin_score = _fraction_value(
                rationale.get("skin_score"),
                f"top_target_rationale[{idx}].rationale.skin_score",
                path,
            )
            top_skin_score_text = str(top_row.get("skin_score", "")).strip()
            if not top_skin_score_text:
                raise ValueError(
                    "iteration manifest top_target_rationale "
                    "rationale.skin_score requires ranking top row column: "
                    f"row index {idx}: {path}"
                )
            top_skin_score = _fraction_value(
                top_skin_score_text,
                (
                    "top_target_rationale ranking top row "
                    f"skin_score for {ranking}"
                ),
                path,
            )
            if not math.isclose(
                skin_score,
                top_skin_score,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "iteration manifest top_target_rationale "
                    "rationale.skin_score must match ranking top row: "
                    f"row index {idx}: {path}"
                )
        for text_col in ("efficacy_category", "cosmetic_decision", "drug_decision"):
            if text_col not in rationale:
                continue
            top_text = str(top_row.get(text_col, "")).strip()
            if not top_text:
                raise ValueError(
                    "iteration manifest top_target_rationale "
                    f"rationale.{text_col} requires ranking top row column: "
                    f"row index {idx}: {path}"
                )
            rationale_text = str(rationale.get(text_col, "")).strip()
            if rationale_text != top_text:
                raise ValueError(
                    "iteration manifest top_target_rationale "
                    f"rationale.{text_col} must match ranking top row: "
                    f"row index {idx}: {path}"
                )


def _validate_iteration_manifest(doc: object, path: Path) -> int:
    if not isinstance(doc, dict):
        raise ValueError(f"iteration manifest must be a JSON object: {path}")

    n_steps = _require_int_field(doc, "n_steps", path)
    n_passed = _require_int_field(doc, "n_passed", path)
    n_failed = _require_int_field(doc, "n_failed", path)
    if n_steps == 0:
        raise ValueError(f"iteration manifest contains no runnable eval steps: {path}")
    if n_failed != 0:
        raise ValueError(f"iteration manifest contains failed eval steps: {path}")
    if n_passed != n_steps:
        raise ValueError(
            "iteration manifest passed step count must equal total steps: "
            f"{path}"
        )
    steps = doc.get("steps")
    if not isinstance(steps, list) or len(steps) != n_steps:
        raise ValueError(
            "iteration manifest steps list length must equal n_steps: "
            f"{path}"
        )
    step_outputs = _validate_eval_steps(steps, path)

    _validate_input_status(doc, path)
    _validate_leakage_status(doc, path)
    threshold_metrics = _validate_threshold_status(doc, path, step_outputs)
    _validate_ranking_metrics(doc, threshold_metrics, path)
    data_digests = _validate_data_snapshot(doc, path)
    _validate_activity_retrieval_gate(doc, path, data_digests)
    _validate_provenance(doc, path, data_digests)
    _validate_manifest_path_bindings(doc, path)
    snapshot_paths = _validate_artifact_snapshot(doc, path, step_outputs)
    _validate_input_runs(doc, path, snapshot_paths)
    _validate_top_target_rationale(doc, path, snapshot_paths)
    if doc.get("claim_ready") is not True:
        raise ValueError(
            f"iteration manifest claim_ready must be true for publication: {path}"
        )
    claim_blockers = doc.get("claim_blockers")
    if not isinstance(claim_blockers, list) or claim_blockers:
        raise ValueError(
            "iteration manifest claim_blockers must be an empty list for "
            f"publication: {path}"
        )
    required_overrides = {
        "allow_incomplete_eval_inputs",
        "allow_empty_iteration",
        "allow_incomplete_leakage",
        "allow_threshold_failure",
        "allow_missing_input_runs",
    }
    diagnostic_overrides = doc.get("diagnostic_overrides")
    if not isinstance(diagnostic_overrides, dict):
        raise ValueError(
            "iteration manifest diagnostic_overrides must be an object for "
            f"publication: {path}"
        )
    override_keys = set(diagnostic_overrides)
    if override_keys != required_overrides:
        missing = sorted(required_overrides - override_keys)
        unexpected = sorted(override_keys - required_overrides)
        raise ValueError(
            "iteration manifest diagnostic_overrides keys must match "
            f"publication contract: missing={missing}, unexpected={unexpected}: "
            f"{path}"
        )
    enabled_overrides = [
        key for key, value in diagnostic_overrides.items() if value is not False
    ]
    if enabled_overrides:
        raise ValueError(
            "iteration manifest diagnostic_overrides must all be false for "
            f"publication: {sorted(enabled_overrides)}: {path}"
        )

    return n_steps


def _validate_pharmacophore_consensus(doc: object, path: Path) -> int:
    if not isinstance(doc, dict) or not doc:
        raise ValueError(
            f"pharmacophore consensus must be a non-empty JSON object: {path}"
        )
    confirmed_count = 0
    for target_id, record in doc.items():
        if not isinstance(target_id, str) or not target_id.strip():
            raise ValueError(
                f"pharmacophore consensus contains a blank target ID: {path}"
            )
        if not isinstance(record, dict):
            raise ValueError(
                "pharmacophore consensus target entries must be objects: "
                f"{target_id}: {path}"
            )
        atoms = record.get("confirmed_atoms")
        if not isinstance(atoms, list):
            raise ValueError(
                "pharmacophore consensus target entry is missing "
                f"confirmed_atoms list: {target_id}: {path}"
            )
        invalid_atom = any(
            isinstance(atom, bool) or not isinstance(atom, int) or atom < 0
            for atom in atoms
        )
        if invalid_atom or len(atoms) != len(set(atoms)):
            raise ValueError(
                "pharmacophore consensus confirmed_atoms must contain unique "
                f"non-negative integers: {target_id}: {path}"
            )
        if not atoms:
            raise ValueError(
                "pharmacophore consensus contains no confirmed atoms for target "
                f"{target_id}: {path}"
            )
        if record.get("claim_eligible") is not True:
            raise ValueError(
                "pose-supported interaction consensus must be claim eligible: "
                f"{target_id}: {path}"
            )
        if record.get("degraded") is not False:
            raise ValueError(
                "pose-supported interaction consensus must not be degraded: "
                f"{target_id}: {path}"
            )
        if record.get("coordinate_system") != "boltz_complex_ligand_atom_order_0_based":
            raise ValueError(
                "pose-supported interaction consensus has an unsupported coordinate system: "
                f"{target_id}: {path}"
            )
        if record.get("evidence_label") != "pose-supported interaction atoms":
            raise ValueError(
                "pose-supported interaction consensus has an unsupported evidence label: "
                f"{target_id}: {path}"
            )
        if record.get("evidence_sources") != ["plip", "prolif"]:
            raise ValueError(
                "pose-supported interaction consensus requires PLIP and ProLIF sources: "
                f"{target_id}: {path}"
            )
        if record.get("consensus_min_votes") != 2:
            raise ValueError(
                "pose-supported interaction consensus requires a 2-of-2 threshold: "
                f"{target_id}: {path}"
            )
        if record.get("pose_supported_interaction_atoms") != atoms:
            raise ValueError(
                "pose-supported interaction atom aliases disagree: "
                f"{target_id}: {path}"
            )
        votes = record.get("votes")
        if not isinstance(votes, dict) or set(votes) != {"plip", "prolif"}:
            raise ValueError(
                "pose-supported interaction consensus votes require PLIP and ProLIF: "
                f"{target_id}: {path}"
            )
        normalized_votes: dict[str, set[int]] = {}
        for source in ("plip", "prolif"):
            source_atoms = votes[source]
            if (
                not isinstance(source_atoms, list)
                or any(
                    isinstance(atom, bool) or not isinstance(atom, int) or atom < 0
                    for atom in source_atoms
                )
                or len(source_atoms) != len(set(source_atoms))
            ):
                raise ValueError(
                    "pose-supported interaction source votes must contain unique "
                    f"non-negative integers: {target_id}:{source}: {path}"
                )
            normalized_votes[source] = set(source_atoms)
        if set(atoms) != normalized_votes["plip"] & normalized_votes["prolif"]:
            raise ValueError(
                "pose-supported interaction consensus does not equal the PLIP/ProLIF "
                f"intersection: {target_id}: {path}"
            )
        confirmed_count += len(atoms)
    if confirmed_count == 0:
        raise ValueError(
            f"pharmacophore consensus contains no confirmed atoms: {path}"
        )
    return confirmed_count


def _validate_eval_manifest_covers_run(
    manifest_path: Path,
    run_dir: Path,
) -> None:
    try:
        doc = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        raise ValueError(
            f"failed to parse iteration manifest for run coverage: {manifest_path}: {exc}"
        ) from exc
    if not isinstance(doc, dict):
        raise ValueError(f"iteration manifest must be a JSON object: {manifest_path}")
    input_runs = doc.get("input_runs")
    runs = input_runs.get("runs") if isinstance(input_runs, dict) else None
    expected = run_dir.expanduser().resolve(strict=False)
    covered = False
    if isinstance(runs, list):
        for run in runs:
            if not isinstance(run, dict):
                continue
            raw_run_dir = run.get("run_dir")
            if not isinstance(raw_run_dir, str) or not raw_run_dir.strip():
                continue
            candidates = _existing_directory_candidates(raw_run_dir, manifest_path)
            if expected in candidates:
                covered = True
                break
    if not covered:
        raise ValueError(
            "iteration manifest input_runs must include the active --run-dir: "
            f"{run_dir}: {manifest_path}"
        )


def validate_source_artifacts(sources: list[Path], run_dir: Path) -> None:
    invalid: list[str] = []
    for source in sources:
        try:
            _source_metric(source)
            if source.name == "iteration_manifest.json":
                _validate_eval_manifest_covers_run(source, run_dir)
        except ValueError as exc:
            invalid.append(str(exc))
    if invalid:
        preview = "; ".join(invalid[:8])
        raise SystemExit(
            "Stage 11 figure package has invalid source artifact(s); "
            f"{preview}"
        )


def _write_source_svg(path: Path, title: str, sources: list[Path]) -> None:
    rows = []
    for src in sources:
        metric, value = _source_metric(src)
        rows.append(f"{src.name}: {value} {metric}")
    text = "\n".join(rows)
    escaped_title = html.escape(title)
    escaped_text = html.escape(text)
    path.write_text(
        "\n".join([
            '<svg xmlns="http://www.w3.org/2000/svg" '
            'width="900" height="420" viewBox="0 0 900 420">',
            '<rect width="900" height="420" fill="#f8fafc"/>',
            '<text x="36" y="54" font-family="Arial, sans-serif" '
            f'font-size="24" font-weight="700">{escaped_title}</text>',
            '<text x="36" y="96" font-family="Arial, sans-serif" '
            'font-size="15" fill="#334155">Diagnostic artifact-count panel '
            'generated from verified upstream artifacts.</text>',
            '<text x="36" y="138" font-family="monospace" font-size="14" '
            f'fill="#0f172a">{escaped_text}</text>',
            "</svg>",
        ])
    )


def _write_source_plot(path: Path, title: str, sources: list[Path]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for claim-quality publication figures; "
            "use --allow-placeholders only for draft diagnostics"
        ) from exc

    labels: list[str] = []
    values: list[int] = []
    for src in sources:
        metric, value = _source_metric(src)
        labels.append(f"{src.name}\n{metric}")
        values.append(max(value, 1))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.barh(labels, values, color="#2f6f73")
    ax.set_title(title)
    ax.set_xlabel("artifact evidence count")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def _write_diagnostic_figure(path: Path, title: str, sources: list[Path]) -> None:
    if path.suffix.lower() == ".svg":
        _write_source_svg(path, title, sources)
    else:
        _write_source_plot(path, title, sources)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--eval-dir", default=Path("results/eval"), type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--out-captions", required=True, type=Path)
    parser.add_argument(
        "--allow-placeholders",
        action="store_true",
        help="emit scaffold placeholder figures for explicit draft diagnostics",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    remove_outputs(expected_outputs(args.out_dir, args.out_captions))
    sources = required_sources(args.run_dir, args.eval_dir)
    missing = [
        p for p in sources if not p.exists() or p.stat().st_size == 0
    ]
    if missing and not args.allow_placeholders:
        preview = ", ".join(str(p) for p in missing[:8])
        raise SystemExit(
            f"Stage 11 figure package requires upstream/eval artifacts; "
            f"missing {len(missing)} file(s). Use --allow-placeholders only for drafts: {preview}"
        )
    if not args.allow_placeholders:
        validate_source_artifacts(sources, args.run_dir)

    args.out_dir.parent.mkdir(parents=True, exist_ok=True)
    captions: dict[str, dict[str, object]] = {}
    with tempfile.TemporaryDirectory(prefix=".stage11_figures_", dir=args.out_dir.parent) as tmp:
        stage_dir = Path(tmp)
        for fname, caption, rels in FIGURE_DEFINITIONS:
            out_path = stage_dir / fname
            sources = [
                args.eval_dir / rel.split(":", 1)[1]
                if rel.startswith("EVAL:")
                else args.run_dir / rel
                for rel in rels
            ]
            if args.allow_placeholders:
                _matplotlib_placeholder(out_path, caption)
                status = "draft_placeholder"
            else:
                _write_diagnostic_figure(out_path, caption, sources)
                # These panels currently visualize artifact row/key/byte counts,
                # not the scientific quantities named by FIGURE_DEFINITIONS.
                # Preserve them as useful provenance diagnostics, but never label
                # them as publication/claim-quality figures.
                status = "diagnostic_only"
            rendered_caption = (
                caption
                if args.allow_placeholders
                else (
                    f"Artifact evidence counts for the sources behind: {caption} "
                    "(row/key counts per source file, not the plotted quantity "
                    "the intended figure describes)"
                )
            )
            captions[fname] = {
                "caption": rendered_caption,
                "intended_caption": caption,
                "rendered_content": (
                    "draft_placeholder"
                    if args.allow_placeholders
                    else "artifact_evidence_counts"
                ),
                "status": status,
                "diagnostic_only": True,
                "claim_eligible": False,
                "sources": [
                    {"path": str(p)}
                    if args.allow_placeholders
                    else _source_evidence(p)
                    for p in sources
                ],
            }
            LOG.info("Prepared %s", out_path)

        stage_captions = stage_dir / args.out_captions.name
        stage_captions.write_text(json.dumps(captions, indent=2))
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for fname, _, _ in FIGURE_DEFINITIONS:
            (stage_dir / fname).replace(args.out_dir / fname)
        args.out_captions.parent.mkdir(parents=True, exist_ok=True)
        stage_captions.replace(args.out_captions)
    LOG.info("Captions → %s", args.out_captions)


if __name__ == "__main__":
    main()
