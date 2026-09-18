#!/usr/bin/env python3
"""stage11_repro_pack.py — Emit a reproducibility pack for the run."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import logging
import math
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

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
    "code_snapshot.json",
    "effective_config.json",
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
CODE_SNAPSHOT_SCHEMA = "skinscout.stage11-code-snapshot.v1"
EFFECTIVE_CONFIG_SCHEMA = "skinscout.stage11-effective-config.v1"
UNTRACKED_SOURCE_SUFFIXES = frozenset(
    {".py", ".smk", ".sh", ".bash", ".yml", ".yaml", ".toml", ".cfg"}
)


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
    return _validate_git_commit_text(commit)


def _validate_git_commit_text(commit: str) -> str:
    if len(commit) != 40 or any(ch not in "0123456789abcdefABCDEF" for ch in commit):
        raise SystemExit(
            "Git commit capture produced an invalid commit SHA; "
            f"expected 40 hex characters, got {commit!r}"
        )
    return commit


def _canonical_sha256(payload: object) -> str:
    data = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _git_bytes(repo_root: Path, *arguments: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", *arguments],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise SystemExit(
            "Code snapshot capture failed; reproducibility pack requires a "
            f"readable git worktree at {repo_root}"
        ) from exc


def _untracked_source_files(repo_root: Path) -> list[dict[str, Any]]:
    """Untracked source files, so a dirty tree is identified by exact bytes.

    Only code/config suffixes are hashed; generated data and run outputs stay
    out of the snapshot. The listing is sorted for a deterministic digest.
    """
    raw = _git_bytes(repo_root, "ls-files", "--others", "--exclude-standard", "-z")
    entries: list[dict[str, Any]] = []
    for relative in raw.decode("utf-8", "surrogateescape").split("\0"):
        if not relative:
            continue
        if Path(relative).suffix.lower() not in UNTRACKED_SOURCE_SUFFIXES:
            continue
        path = repo_root / relative
        if not path.is_file():
            continue
        data = path.read_bytes()
        entries.append(
            {
                "path": relative,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return sorted(entries, key=lambda entry: entry["path"])


def _code_snapshot_record(
    *,
    git_commit: str,
    tracked_diff: bytes,
    untracked_files: list[dict[str, Any]],
    limitation: dict[str, Any] | None,
) -> dict[str, Any]:
    """Seal the executed code bytes into one deterministic digest."""
    commit = _validate_git_commit_text(git_commit)
    payload: dict[str, Any] = {
        "schema_version": CODE_SNAPSHOT_SCHEMA,
        "git_commit": commit,
        "dirty": bool(tracked_diff) or bool(untracked_files),
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "tracked_diff_bytes": len(tracked_diff),
        "untracked_files": untracked_files,
        "untracked_tree_sha256": _canonical_sha256(untracked_files),
        "limitation": limitation,
    }
    payload["snapshot_sha256"] = _canonical_sha256(payload)
    payload["captured_at_utc"] = (
        datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )
    return payload


def _capture_code_snapshot(
    repo_root: Path,
    *,
    limitation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    commit = _git_bytes(repo_root, "rev-parse", "HEAD").decode("ascii", "replace").strip()
    tracked_diff = _git_bytes(repo_root, "diff", "HEAD", "--", ".")
    untracked = _untracked_source_files(repo_root)
    return _code_snapshot_record(
        git_commit=commit,
        tracked_diff=tracked_diff,
        untracked_files=untracked,
        limitation=limitation,
    )


def _load_dirty_snapshot_limitation(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(
            "Dirty snapshot limitation record is required and must be non-empty: "
            f"{path}"
        )
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        raise SystemExit(
            f"Dirty snapshot limitation record failed to parse: {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise SystemExit(
            f"Dirty snapshot limitation record must be a JSON object: {path}"
        )
    reason = payload.get("reason")
    approved_by = payload.get("approved_by")
    if (
        not isinstance(reason, str)
        or not reason.strip()
        or not isinstance(approved_by, str)
        or not approved_by.strip()
    ):
        raise SystemExit(
            "Dirty snapshot limitation record requires non-empty 'reason' and "
            f"'approved_by' strings: {path}"
        )
    record: dict[str, Any] = {
        "reason": reason.strip(),
        "approved_by": approved_by.strip(),
    }
    recorded_at = payload.get("recorded_at_utc")
    if isinstance(recorded_at, str) and recorded_at.strip():
        record["recorded_at_utc"] = recorded_at.strip()
    return record


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


def _apply_config_override(config: dict[str, Any], override: str) -> None:
    key, separator, raw = override.partition("=")
    if not separator or not key.strip():
        raise SystemExit(
            f"Config override must use KEY=VALUE syntax: {override!r}"
        )
    parts = [part.strip() for part in key.strip().split(".")]
    if any(not part for part in parts):
        raise SystemExit(
            f"Config override contains an empty key segment: {override!r}"
        )
    try:
        value = yaml.safe_load(raw) if raw.strip() else ""
    except yaml.YAMLError as exc:
        raise SystemExit(
            f"Config override value failed to parse: {override!r}: {exc}"
        ) from exc
    cursor = config
    for part in parts[:-1]:
        child = cursor.setdefault(part, {})
        if not isinstance(child, dict):
            raise SystemExit(
                "Config override conflicts with a scalar key: "
                f"{override!r}"
            )
        cursor = child
    cursor[parts[-1]] = value


def _effective_config_digest(
    config_path: Path,
    overrides: list[str],
) -> str:
    """Canonical digest of the default YAML merged with CLI overrides.

    Hashing only ``workflow/config.yaml`` lets two runs with different
    ``--config`` overrides share a digest, so the effective object is hashed.
    """
    try:
        config = yaml.safe_load(config_path.read_text())
    except yaml.YAMLError as exc:
        raise SystemExit(
            f"Workflow config failed to parse for effective digest: {config_path}: {exc}"
        ) from exc
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise SystemExit(
            f"Workflow config must be a mapping for effective digest: {config_path}"
        )
    for override in overrides:
        _apply_config_override(config, override)
    return _canonical_sha256(config)


def _effective_config_record(
    *,
    config_path: Path,
    config_sha256: str,
    overrides: list[str],
    run_dir: Path,
) -> dict[str, Any]:
    """Bind the Stage 11 config sidecar to the run's effective configuration.

    ``run_manifest.json`` is sealed at run start and its ``hashes.config_sha256``
    canonicalizes the default YAML digest, the run profile, the target list and
    the Snakemake ``--config`` override list. When it is available it is the
    authoritative effective-config identity; otherwise the digest is recomputed
    from the default YAML plus the recorded CLI overrides.
    """
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.is_file() and manifest_path.stat().st_size > 0:
        try:
            manifest = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
            raise SystemExit(
                f"Run manifest failed to parse for effective config: {manifest_path}: {exc}"
            ) from exc
        hashes = manifest.get("hashes") if isinstance(manifest, dict) else None
        digest = hashes.get("config_sha256") if isinstance(hashes, dict) else None
        if not isinstance(digest, str) or len(digest) != 64:
            raise SystemExit(
                f"Run manifest hashes.config_sha256 is invalid: {manifest_path}"
            )
        return {
            "schema_version": EFFECTIVE_CONFIG_SCHEMA,
            "source": "run_manifest",
            "base_config_sha256": config_sha256,
            "effective_config_sha256": digest.lower(),
            "run_manifest_path": str(manifest_path),
            "overrides": list(overrides),
        }
    return {
        "schema_version": EFFECTIVE_CONFIG_SCHEMA,
        "source": "config_yaml_plus_cli_overrides",
        "base_config_sha256": config_sha256,
        "effective_config_sha256": _effective_config_digest(
            config_path, overrides
        ),
        "overrides": list(overrides),
    }


def _producer_literal_int(producer: Path, name: str) -> int:
    """Read a seed constant the producer implementation itself consumes."""
    try:
        tree = ast.parse(producer.read_text(), filename=str(producer))
    except (OSError, SyntaxError) as exc:
        raise SystemExit(
            f"Could not read producer constant {name} from {producer}: {exc}"
        ) from exc
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == name
            for target in statement.targets
        ):
            continue
        try:
            value = ast.literal_eval(statement.value)
        except (ValueError, TypeError) as exc:
            raise SystemExit(
                f"{name} in {producer} must be a literal non-negative integer"
            ) from exc
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SystemExit(
                f"{name} in {producer} must be a literal non-negative integer"
            )
        return value
    raise SystemExit(f"{producer} does not declare {name}")


def _effective_etkdg_seed() -> int:
    """Read the seed constant used by the Stage 1 producer implementation."""
    producer = Path(__file__).with_name("stage1_etkdg.py")
    return _producer_literal_int(producer, "ETKDG_RANDOM_SEED")


def _etkdg_seed_metadata(run_dir: Path) -> dict[str, Any]:
    conformers = run_dir / "01_structure" / "conformers_etkdg.sdf"
    current_default = _effective_etkdg_seed()
    if not conformers.is_file() or conformers.stat().st_size == 0:
        return {
            "seed": None,
            "observed": False,
            "source": "run_artifact_unavailable",
            "current_producer_default": current_default,
        }
    lines = conformers.read_text(errors="replace").splitlines()
    values: list[int] = []
    for index, line in enumerate(lines):
        if not line.strip().startswith(">") or "<etkdg_random_seed>" not in line:
            continue
        if index + 1 >= len(lines):
            raise SystemExit(
                f"ETKDG seed property has no value in run artifact: {conformers}"
            )
        raw = lines[index + 1].strip()
        try:
            seed = int(raw)
        except ValueError as exc:
            raise SystemExit(
                f"ETKDG seed property must be a non-negative integer in "
                f"run artifact: {conformers}: {raw!r}"
            ) from exc
        if seed < 0:
            raise SystemExit(
                f"ETKDG seed property must be a non-negative integer in "
                f"run artifact: {conformers}: {raw!r}"
            )
        values.append(seed)
    if not values:
        return {
            "seed": None,
            "observed": False,
            "source": "run_artifact_property_unavailable",
            "artifact": str(conformers),
            "artifact_sha256": _sha256(conformers),
            "current_producer_default": current_default,
        }
    unique = sorted(set(values))
    if len(unique) != 1:
        raise SystemExit(
            f"ETKDG conformers record conflicting random seeds in {conformers}: "
            f"{unique}"
        )
    return {
        "seed": unique[0],
        "observed": True,
        "source": "run_artifact_sdf_property",
        "artifact": str(conformers),
        "artifact_sha256": _sha256(conformers),
    }


SEED_MANIFEST_SCHEMA = "skinscout.stage11-random-seeds.v1"


def _relative_artifact(path: Path, run_dir: Path) -> str:
    try:
        return str(path.relative_to(run_dir))
    except ValueError:
        return str(path)


def _seed_control_record(
    *,
    stage: str,
    backend: str,
    algorithm: str,
    control: str,
    source: str,
    seed: int | None = None,
    replica_seeds: list[int] | None = None,
    artifact: Path | None = None,
    run_dir: Path | None = None,
    **extra: Any,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "stage": stage,
        "backend": backend,
        "algorithm": algorithm,
        "control": control,
        "seed": seed,
        "source": source,
    }
    if replica_seeds is not None:
        record["replica_seeds"] = replica_seeds
    if artifact is not None:
        record["artifact"] = (
            _relative_artifact(artifact, run_dir) if run_dir is not None else str(artifact)
        )
        record["artifact_sha256"] = _sha256(artifact)
    record.update(extra)
    return record


def _unavailable_seed_record(
    *,
    stage: str,
    backend: str,
    algorithm: str,
    artifact: Path | None = None,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    return _seed_control_record(
        stage=stage,
        backend=backend,
        algorithm=algorithm,
        control="unavailable",
        source="run_artifact_absent",
        seed=None,
        artifact=(
            artifact
            if artifact is not None and artifact.is_file() and artifact.stat().st_size > 0
            else None
        ),
        run_dir=run_dir,
        reason="no run artifact recorded this stage in the sealed run",
    )


def _read_seed_column(path: Path, run_dir: Path, column: str) -> list[int]:
    """Read a per-row seed column a producer wrote; fail closed if malformed."""
    rel = _relative_artifact(path, run_dir)
    try:
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            fieldnames = reader.fieldnames or []
            rows = list(reader)
    except (csv.Error, UnicodeDecodeError, OSError) as exc:
        raise SystemExit(
            f"Stage seed artifact failed to parse as a table: {rel}: {exc}"
        ) from exc
    if column not in fieldnames:
        raise SystemExit(
            f"Stage seed artifact {rel} lacks the '{column}' column needed "
            "for seed provenance"
        )
    seeds: list[int] = []
    for idx, row in enumerate(rows):
        raw = str(row.get(column, "")).strip()
        if not raw:
            raise SystemExit(
                f"Stage seed artifact column '{column}' is blank at row "
                f"index {idx}: {rel}"
            )
        try:
            seed = int(raw)
        except ValueError as exc:
            raise SystemExit(
                f"Stage seed artifact column '{column}' must be an integer at "
                f"row index {idx}: {rel}: {raw!r}"
            ) from exc
        if seed < 0:
            raise SystemExit(
                f"Stage seed artifact column '{column}' must be a "
                f"non-negative integer at row index {idx}: {rel}: {raw!r}"
            )
        seeds.append(seed)
    if not seeds:
        raise SystemExit(f"Stage seed artifact contains no data rows: {rel}")
    return seeds


def _json_seed_record(
    path: Path,
    run_dir: Path,
    *,
    stage: str,
    backend: str,
    algorithm: str,
) -> dict[str, Any]:
    """Record the seed a stage manifest says the producer actually used."""
    if not path.is_file() or path.stat().st_size == 0:
        return _unavailable_seed_record(
            stage=stage, backend=backend, algorithm=algorithm, artifact=path, run_dir=run_dir
        )
    try:
        doc = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        raise SystemExit(
            f"Stage seed manifest failed to parse as JSON: "
            f"{_relative_artifact(path, run_dir)}: {exc}"
        ) from exc
    if not isinstance(doc, dict) or "seed" not in doc:
        raise SystemExit(
            f"Stage seed manifest must be an object carrying 'seed': "
            f"{_relative_artifact(path, run_dir)}"
        )
    raw_seed = doc["seed"]
    if isinstance(raw_seed, bool) or not isinstance(raw_seed, int) or raw_seed < 0:
        raise SystemExit(
            f"Stage seed manifest seed must be a non-negative integer: "
            f"{_relative_artifact(path, run_dir)}: {raw_seed!r}"
        )
    return _seed_control_record(
        stage=stage,
        backend=backend,
        algorithm=algorithm,
        control="seeded",
        seed=raw_seed,
        source="run_artifact_manifest",
        artifact=path,
        run_dir=run_dir,
        execution_status=doc.get("execution_status"),
    )


def _stage_seed_payload(run_dir: Path) -> dict[str, Any]:
    """Per-stage seed provenance from the artifacts each producer wrote.

    Environment variables are deliberately not consulted: the old payload
    recorded `BOLTZ_SEED`/`REINVENT_SEED` even though no producer reads them,
    and a seed a stage did not consume must never be published as if it had
    controlled the result. Uncontrollable stochastic stages are marked
    `uncontrolled`; stages absent from this run are `unavailable`.
    """
    stages: dict[str, Any] = {}

    etkdg = _etkdg_seed_metadata(run_dir)
    stages["etkdg"] = {
        **etkdg,
        "stage": "stage1_etkdg",
        "backend": "RDKit",
        "algorithm": "ETKDGv3 conformer embedding",
        "control": "seeded" if etkdg.get("observed") else "unavailable",
    }

    stages["boltz2"] = _seed_control_record(
        stage="stage5_boltz2",
        backend="Boltz-2",
        algorithm="diffusion co-folding and affinity",
        control="uncontrolled",
        source="producer_has_no_seed_option",
        producer="scripts/stage5_boltz2.py",
    )
    stages["bioemu"] = _seed_control_record(
        stage="stage6_bioemu",
        backend="BioEmu",
        algorithm="sequence-conditioned diffusion sampling",
        control="uncontrolled",
        source="producer_has_no_seed_option",
        producer="scripts/stage6_bioemu.py",
    )

    ensemble_manifest = run_dir / "06_bioemu" / "ensemble_manifest.tsv"
    if ensemble_manifest.is_file() and ensemble_manifest.stat().st_size > 0:
        sidechain_seed = _producer_literal_int(
            Path(__file__).with_name("stage6_bioemu.py"), "RECONSTRUCTION_SEED"
        )
        stages["sidechain_reconstruction"] = _seed_control_record(
            stage="stage6_sidechain_reconstruction",
            backend="pdbfixer + OpenMM",
            algorithm="missing-atom placement and constrained minimisation",
            control="seeded",
            seed=sidechain_seed,
            source="producer_constant",
            producer="scripts/stage6_bioemu.py",
            artifact=ensemble_manifest,
            run_dir=run_dir,
        )
    else:
        stages["sidechain_reconstruction"] = _unavailable_seed_record(
            stage="stage6_sidechain_reconstruction",
            backend="pdbfixer + OpenMM",
            algorithm="missing-atom placement and constrained minimisation",
        )

    consensus = run_dir / "06_bioemu" / "ensemble_consensus.tsv"
    if consensus.is_file() and consensus.stat().st_size > 0:
        docking_seeds = sorted(set(_read_seed_column(consensus, run_dir, "docking_seed")))
        if len(docking_seeds) != 1:
            raise SystemExit(
                "Ensemble docking consensus records conflicting docking seeds: "
                f"{docking_seeds}: {_relative_artifact(consensus, run_dir)}"
            )
        stages["ensemble_dock"] = _seed_control_record(
            stage="stage6_ensemble_dock",
            backend="GNINA",
            algorithm="ensemble docking with consensus ranking",
            control="seeded",
            seed=docking_seeds[0],
            source="run_artifact_ensemble_consensus_tsv",
            artifact=consensus,
            run_dir=run_dir,
        )
    else:
        stages["ensemble_dock"] = _unavailable_seed_record(
            stage="stage6_ensemble_dock",
            backend="GNINA",
            algorithm="ensemble docking with consensus ranking",
        )

    trajectory_index = run_dir / "07_md" / "trajectory_index.tsv"
    if trajectory_index.is_file() and trajectory_index.stat().st_size > 0:
        replica_seeds = sorted(set(_read_seed_column(trajectory_index, run_dir, "seed")))
        stages["gromacs_production"] = _seed_control_record(
            stage="stage7_gromacs_production",
            backend="GROMACS",
            algorithm="Langevin MD replicas",
            control="seeded",
            seed=replica_seeds[0],
            replica_seeds=replica_seeds,
            source="run_artifact_trajectory_index_tsv",
            artifact=trajectory_index,
            run_dir=run_dir,
        )
    else:
        stages["gromacs_production"] = _unavailable_seed_record(
            stage="stage7_gromacs_production",
            backend="GROMACS",
            algorithm="Langevin MD replicas",
        )

    stages["reinvent4_finetune"] = _json_seed_record(
        run_dir / "05_6_analogs" / "reinvent4_finetune_manifest.json",
        run_dir,
        stage="stage5_6_reinvent4_finetune",
        backend="REINVENT4",
        algorithm="transfer learning",
    )
    stages["reinvent4_generation"] = _json_seed_record(
        run_dir / "05_6_analogs" / "reinvent4_status.json",
        run_dir,
        stage="stage5_6_reinvent4_generation",
        backend="REINVENT4",
        algorithm="reinvent sampling",
    )

    return {
        "schema_version": SEED_MANIFEST_SCHEMA,
        "run_dir": str(run_dir),
        "stages": stages,
    }


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
    parser.add_argument(
        "--config-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Snakemake-style config override applied on top of the default YAML "
            "when no sealed run manifest is available; repeat as needed"
        ),
    )
    parser.add_argument(
        "--dirty-snapshot-limitation",
        default=None,
        type=Path,
        help="JSON record (reason, approved_by) authorizing a dirty code snapshot",
    )
    parser.add_argument("--eval-manifest", default=None, type=Path)
    parser.add_argument("--fast-report-pointer", default=None, type=Path)
    parser.add_argument("--physics-report-pointer", default=None, type=Path)
    parser.add_argument("--require-physics-report", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    _remove_outputs(args.out_dir)
    # Seal the executed code bytes before any other work so a mid-run edit
    # cannot be recorded as the source of this pack.
    repo_root = Path(__file__).resolve().parents[1]
    limitation = _load_dirty_snapshot_limitation(args.dirty_snapshot_limitation)
    code_snapshot = _capture_code_snapshot(repo_root, limitation=limitation)
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
    effective_config = _effective_config_record(
        config_path=args.config,
        config_sha256=config_sha256,
        overrides=list(args.config_override),
        run_dir=args.run_dir,
    )
    git_commit = str(code_snapshot["git_commit"])
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
        "code_snapshot.json": json.dumps(code_snapshot, indent=2, sort_keys=True) + "\n",
        "effective_config.json": json.dumps(
            effective_config, indent=2, sort_keys=True
        ) + "\n",
        "random_seeds.json": json.dumps(
            _stage_seed_payload(args.run_dir), indent=2
        ) + "\n",
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
