#!/usr/bin/env python3
"""Manifest-driven Stage 5.6 REINVENT4 adapter.

This adapter intentionally does not invent REINVENT4 command-line flags. The
plugin manifest must provide the executable arguments and declare where the raw
generated output is written. The adapter validates the configured executable,
model artifacts, config, plugin manifest, and upstream inputs before execution;
then it normalizes generated molecules to canonical SMILES and writes lineage
records.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from interaction_anchor import InteractionAnchorError, load_anchor_map


STAGE = "5.6"
STATUS_BLOCKED = "blocked"
STATUS_COMPLETED = "completed"


class AdapterError(RuntimeError):
    """Fatal adapter error that must leave no stale outputs."""


class Blocked(AdapterError):
    """Missing dependency or artifact gate."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path, label: str) -> Any:
    require_nonempty_file(path, label)
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise AdapterError(f"{label} is not valid JSON: {path}: {exc}") from exc


def require_nonempty_file(path: Path, label: str) -> None:
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        raise Blocked(f"{label} is required and must be a non-empty file: {path}")


def require_executable(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise Blocked(f"REINVENT executable is missing: {path}")
    if not os.access(path, os.X_OK):
        raise Blocked(f"REINVENT executable is not executable: {path}")


def resolve_manifest_path(value: str, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def validate_input_sdf(path: Path) -> str:
    require_nonempty_file(path, "Input SDF")
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise Blocked("RDKit is required to validate input SDF and SMILES") from exc

    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    for molecule in supplier:
        if molecule is not None:
            return Chem.MolToSmiles(molecule, canonical=True)
    raise AdapterError(f"Input SDF contains no readable molecule: {path}")


def validate_consensus(path: Path) -> dict[str, Any]:
    payload = read_json(path, "Pose-supported interaction consensus")
    if not isinstance(payload, dict) or not payload:
        raise AdapterError(f"Pose-supported interaction consensus must be a non-empty JSON object: {path}")
    return payload


def validate_config(path: Path) -> str:
    require_nonempty_file(path, "REINVENT config")
    suffix = path.suffix.lower()
    if suffix == ".json":
        read_json(path, "REINVENT config")
    elif suffix == ".toml":
        try:
            import tomllib
        except ImportError as exc:
            raise Blocked("tomllib is required to validate REINVENT TOML config") from exc
        try:
            tomllib.loads(path.read_text())
        except tomllib.TOMLDecodeError as exc:
            raise AdapterError(f"REINVENT config is not valid TOML: {path}: {exc}") from exc
    return sha256_file(path)


def validate_model_manifest(path: Path) -> list[dict[str, str]]:
    payload = read_json(path, "REINVENT model manifest")
    if not isinstance(payload, Mapping):
        raise AdapterError(f"REINVENT model manifest must be a JSON object: {path}")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise Blocked(f"REINVENT model manifest must list at least one artifact: {path}")
    base = path.parent
    normalized: list[dict[str, str]] = []
    for idx, item in enumerate(artifacts):
        if isinstance(item, str):
            artifact_path = resolve_manifest_path(item, base)
            label = item
            expected_hash = None
        elif isinstance(item, Mapping):
            raw_path = item.get("path")
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise Blocked(f"REINVENT model artifact {idx} is missing a path: {path}")
            artifact_path = resolve_manifest_path(raw_path, base)
            label = str(item.get("name") or raw_path)
            expected_hash = item.get("sha256")
        else:
            raise AdapterError(f"REINVENT model artifact entries must be strings or objects: {path}")
        require_nonempty_file(artifact_path, f"REINVENT model artifact {label}")
        actual_hash = sha256_file(artifact_path)
        if expected_hash is not None and actual_hash != expected_hash:
            raise Blocked(
                f"REINVENT model artifact hash mismatch for {label}: "
                f"expected {expected_hash}, observed {actual_hash}"
            )
        normalized.append({"label": label, "path": str(artifact_path), "sha256": actual_hash})
    return normalized


def validate_plugin_manifest(path: Path) -> dict[str, Any]:
    payload = read_json(path, "REINVENT plugin manifest")
    if not isinstance(payload, Mapping):
        raise AdapterError(f"REINVENT plugin manifest must be a JSON object: {path}")
    command_args = payload.get("command_args")
    if not isinstance(command_args, list) or not command_args:
        raise Blocked(f"REINVENT plugin manifest must define non-empty command_args: {path}")
    if not all(isinstance(value, str) for value in command_args):
        raise AdapterError(f"REINVENT plugin manifest command_args must be strings: {path}")
    joined = "\n".join(command_args)
    if "{seed}" not in joined:
        raise Blocked("REINVENT plugin manifest command_args must propagate the deterministic {seed}")
    if "{generated_output}" not in joined:
        raise Blocked("REINVENT plugin manifest command_args must declare the {generated_output} path")
    if "{interaction_anchors}" not in joined:
        raise Blocked(
            "REINVENT plugin manifest command_args must propagate the "
            "target-conditioned {interaction_anchors} path"
        )
    required_files = payload.get("required_files", [])
    if not isinstance(required_files, list):
        raise AdapterError(f"REINVENT plugin manifest required_files must be a list: {path}")
    for idx, item in enumerate(required_files):
        if not isinstance(item, str) or not item.strip():
            raise AdapterError(f"REINVENT plugin manifest required_files[{idx}] must be a path string: {path}")
        require_nonempty_file(resolve_manifest_path(item, path.parent), f"REINVENT plugin required file {item}")
    return dict(payload)


def format_args(command_args: list[str], placeholders: Mapping[str, str]) -> list[str]:
    formatted: list[str] = []
    for arg in command_args:
        try:
            formatted.append(arg.format_map(placeholders))
        except KeyError as exc:
            raise AdapterError(f"Unknown placeholder in plugin command_args: {exc.args[0]}") from exc
    return formatted


def canonical_smiles(smiles: str) -> str:
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise Blocked("RDKit is required to normalize generated SMILES") from exc
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise AdapterError(f"Generated output contains invalid SMILES: {smiles}")
    return Chem.MolToSmiles(molecule, canonical=True)


def smiles_from_json(payload: Any) -> Iterable[tuple[str, str]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, Mapping):
        rows = payload.get("molecules") or payload.get("generated") or payload.get("smiles") or []
    else:
        rows = []
    if isinstance(rows, str):
        yield rows, "json"
        return
    if not isinstance(rows, list):
        return
    for row in rows:
        if isinstance(row, str):
            yield row, "json"
        elif isinstance(row, Mapping):
            value = row.get("smiles") or row.get("SMILES")
            if value is not None:
                yield str(value), "json"


def parse_generated_text(text: str) -> list[tuple[str, str]]:
    stripped = text.strip()
    if not stripped:
        return []
    if stripped[0] in "[{":
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = None
        if parsed is not None:
            return list(smiles_from_json(parsed))

    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    first = lines[0].strip()
    delimiter = "," if "," in first else "\t" if "\t" in first else None
    if delimiter is not None:
        reader = csv.DictReader(lines, delimiter=delimiter)
        fieldnames = reader.fieldnames or []
        smiles_column = next((name for name in fieldnames if name.lower() == "smiles"), None)
        if smiles_column is not None:
            return [(str(row[smiles_column]), "table") for row in reader if row.get(smiles_column)]

    out: list[tuple[str, str]] = []
    for line in lines:
        token = line.strip().split()[0]
        if token.lower() == "smiles":
            continue
        out.append((token, "line"))
    return out


def normalize_generated(generated_path: Path, stdout: str) -> list[dict[str, Any]]:
    raw_parts: list[tuple[str, str]] = []
    if generated_path.exists() and generated_path.stat().st_size > 0:
        raw_parts.extend(parse_generated_text(generated_path.read_text()))
    raw_parts.extend(parse_generated_text(stdout))

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source_index, (raw_smiles, source_kind) in enumerate(raw_parts):
        raw = raw_smiles.strip()
        if not raw:
            continue
        canonical = canonical_smiles(raw)
        if canonical in seen:
            continue
        seen.add(canonical)
        rows.append(
            {
                "analog_id": f"reinvent4_{len(rows) + 1:06d}",
                "smiles": canonical,
                "raw_smiles": raw,
                "source_kind": source_kind,
                "source_index": source_index,
                "smiles_sha256": sha256_text(canonical),
            }
        )
    return rows


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(text)
    tmp.replace(path)


def write_lineage_csv(path: Path, rows: list[dict[str, Any]], *, seed: int) -> None:
    fieldnames = [
        "analog_id",
        "smiles",
        "raw_smiles",
        "source_kind",
        "source_index",
        "seed",
        "smiles_sha256",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "seed": seed})
    tmp.replace(path)


def output_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    return {label: sha256_file(path) for label, path in paths.items() if path.exists()}


def remove_outputs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


def write_blocked_outputs(args: argparse.Namespace, reason: str, input_smiles: str | None = None) -> None:
    payload = {
        "schema_version": 1,
        "stage": STAGE,
        "component": "reinvent4_adapter",
        "execution_status": STATUS_BLOCKED,
        "claim_ready": False,
        "claim_eligible": False,
        "diagnostic_only": True,
        "blocker": reason,
        "seed": args.seed,
        "input_smiles": input_smiles,
    }
    atomic_write_text(args.out_smi, "")
    write_lineage_csv(args.out_lineage_csv, [], seed=args.seed)
    atomic_write_text(args.out_lineage_json, json.dumps({"schema_version": 1, "molecules": []}, indent=2) + "\n")
    atomic_write_text(args.out_stdout, "")
    atomic_write_text(args.out_stderr, "")
    atomic_write_text(args.out_status, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_adapter(args: argparse.Namespace) -> None:
    outputs = [
        args.out_smi,
        args.out_lineage_csv,
        args.out_lineage_json,
        args.out_status,
        args.out_stdout,
        args.out_stderr,
    ]
    remove_outputs(outputs)

    input_smiles: str | None = None
    try:
        require_executable(args.executable)
        input_smiles = validate_input_sdf(args.in_sdf)
        consensus = validate_consensus(args.consensus)
        try:
            interaction_anchors = load_anchor_map(
                args.interaction_anchors,
                parent_sdf=args.in_sdf,
                consensus_json=args.consensus,
                boltz_report=args.boltz_report,
                verify_source_files=True,
            )
        except InteractionAnchorError as exc:
            raise AdapterError(str(exc)) from exc
        anchor_targets = sorted(interaction_anchors["targets"])
        if not set(anchor_targets).issubset(consensus):
            raise AdapterError(
                "Interaction-anchor targets must be present in the pose-supported consensus"
            )
        config_hash = validate_config(args.config)
        model_artifacts = validate_model_manifest(args.model_manifest)
        plugin = validate_plugin_manifest(args.plugin_manifest)

        output_root = args.out_status.parent
        output_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="stage5_6_reinvent.", dir=output_root) as tmpdir:
            generated_output = Path(tmpdir) / "generated.raw"
            placeholders = {
                "config": str(args.config),
                "input_sdf": str(args.in_sdf),
                "consensus": str(args.consensus),
                "interaction_anchors": str(args.interaction_anchors),
                "generated_output": str(generated_output),
                "seed": str(args.seed),
                "model_manifest": str(args.model_manifest),
                "plugin_manifest": str(args.plugin_manifest),
                "first_model_artifact": model_artifacts[0]["path"],
            }
            command = [str(args.executable), *format_args(plugin["command_args"], placeholders)]
            started = time.time()
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=args.timeout_seconds,
            )
            elapsed = time.time() - started
            if result.returncode != 0:
                raise AdapterError(
                    f"REINVENT executable failed with exit code {result.returncode}; "
                    f"stderr: {result.stderr.strip()[:500]}"
                )
            rows = normalize_generated(generated_output, result.stdout)
            if not rows:
                raise AdapterError("REINVENT executable produced no generated SMILES")

            smiles_text = "".join(f"{row['smiles']}\t{row['analog_id']}\n" for row in rows)
            lineage_payload = {
                "schema_version": 1,
                "stage": STAGE,
                "component": "reinvent4_adapter",
                "seed": args.seed,
                "input_smiles": input_smiles,
                "consensus_targets": sorted(consensus),
                "interaction_anchor_targets": anchor_targets,
                "interaction_anchor_sha256": sha256_file(args.interaction_anchors),
                "molecules": [{**row, "seed": args.seed} for row in rows],
            }
            status_payload = {
                "schema_version": 1,
                "stage": STAGE,
                "component": "reinvent4_adapter",
                "execution_status": STATUS_COMPLETED,
                "claim_ready": False,
                "claim_eligible": False,
                "diagnostic_only": False,
                "seed": args.seed,
                "input_smiles": input_smiles,
                "interaction_anchor_targets": anchor_targets,
                "interaction_anchor_sha256": sha256_file(args.interaction_anchors),
                "generated_count": len(rows),
                "command": command,
                "returncode": result.returncode,
                "elapsed_seconds": elapsed,
                "model_artifacts": model_artifacts,
                "config_sha256": config_hash,
                "plugin_manifest_sha256": sha256_file(args.plugin_manifest),
                "stdout_sha256": sha256_text(result.stdout),
                "stderr_sha256": sha256_text(result.stderr),
            }

            atomic_write_text(args.out_smi, smiles_text)
            write_lineage_csv(args.out_lineage_csv, rows, seed=args.seed)
            atomic_write_text(args.out_lineage_json, json.dumps(lineage_payload, indent=2, sort_keys=True) + "\n")
            atomic_write_text(args.out_stdout, result.stdout)
            atomic_write_text(args.out_stderr, result.stderr)
            status_payload["output_hashes"] = output_hashes(
                {
                    "smi": args.out_smi,
                    "lineage_csv": args.out_lineage_csv,
                    "lineage_json": args.out_lineage_json,
                    "stdout": args.out_stdout,
                    "stderr": args.out_stderr,
                }
            )
            atomic_write_text(args.out_status, json.dumps(status_payload, indent=2, sort_keys=True) + "\n")
    except Blocked as exc:
        remove_outputs(outputs)
        if not args.allow_empty_output:
            raise SystemExit(str(exc)) from exc
        write_blocked_outputs(args, str(exc), input_smiles=input_smiles)
    except Exception as exc:
        remove_outputs(outputs)
        raise SystemExit(str(exc)) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-sdf", required=True, type=Path)
    parser.add_argument("--consensus", required=True, type=Path)
    parser.add_argument("--interaction-anchors", required=True, type=Path)
    parser.add_argument("--boltz-report", required=True, type=Path)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--plugin-manifest", required=True, type=Path)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--out-smi", required=True, type=Path)
    parser.add_argument("--out-lineage-csv", required=True, type=Path)
    parser.add_argument("--out-lineage-json", required=True, type=Path)
    parser.add_argument("--out-status", required=True, type=Path)
    parser.add_argument("--out-stdout", type=Path)
    parser.add_argument("--out-stderr", type=Path)
    parser.add_argument(
        "--allow-empty-output",
        action="store_true",
        help="For missing dependency/artifact gates only, emit explicit blocked outputs.",
    )
    args = parser.parse_args()
    if args.seed < 0:
        raise SystemExit("--seed must be a non-negative integer")
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    if args.out_stdout is None:
        args.out_stdout = args.out_status.with_suffix(".stdout.txt")
    if args.out_stderr is None:
        args.out_stderr = args.out_status.with_suffix(".stderr.txt")
    if shutil.which(str(args.executable)) is not None and not args.executable.is_absolute():
        args.executable = Path(shutil.which(str(args.executable)) or str(args.executable))
    return args


def main() -> None:
    run_adapter(parse_args())


if __name__ == "__main__":
    main()
