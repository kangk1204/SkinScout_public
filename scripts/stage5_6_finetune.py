#!/usr/bin/env python3
"""Run a reproducible REINVENT4 transfer-learning preparation job.

The REINVENT4 repository is intentionally installed outside SkinScout because
its CUDA/PyTorch build is hardware-specific.  This wrapper owns the data
contract, hold-out exclusion, command execution, and model provenance.  The
operator supplies the exact upstream-compatible command template and output
paths; SkinScout never guesses a REINVENT CLI or silently accepts a partial
model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = 1
DEFAULT_SEED = 49242
PLACEHOLDERS = {
    "train_smi",
    "prior",
    "out_dir",
    "agent",
    "staged_config",
    "plugin_manifest",
    "seed",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_nonempty(path: Path, label: str) -> None:
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"{label} is required and must be non-empty: {path}")


def _read_smiles(path: Path, label: str) -> list[str]:
    _require_nonempty(path, label)
    from rdkit import Chem

    values: list[str] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        smiles = text.split()[0]
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise SystemExit(
                f"{label} contains invalid SMILES at line {line_number}: {path}"
            )
        values.append(Chem.MolToSmiles(molecule, canonical=True))
    if not values:
        raise SystemExit(f"{label} contains no SMILES: {path}")
    if len(values) != len(set(values)):
        raise SystemExit(f"{label} contains duplicate canonical SMILES: {path}")
    return values


def _format_command(template: str, values: dict[str, str]) -> list[str]:
    if not template.strip():
        raise SystemExit("--command-template must be non-empty")
    try:
        rendered = template.format(**values)
    except KeyError as exc:
        raise SystemExit(
            f"--command-template contains unsupported placeholder: {exc.args[0]}"
        ) from exc
    if "{" in rendered or "}" in rendered:
        raise SystemExit(
            "--command-template contains an unresolved brace; use only the supported "
            f"placeholders: {', '.join(sorted(PLACEHOLDERS))}"
        )
    command = shlex.split(rendered)
    if not command:
        raise SystemExit("--command-template produced an empty command")
    return command


def _artifact(path: Path, label: str) -> dict[str, object]:
    _require_nonempty(path, label)
    return {
        "label": label,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _remove_output(path: Path) -> None:
    path.unlink(missing_ok=True)


def _blocked_payload(args: argparse.Namespace, blockers: Iterable[str]) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "5.6",
        "component": "reinvent4_transfer_learning",
        "execution_status": "blocked",
        "claim_eligible": False,
        "diagnostic_only": True,
        "seed": args.seed,
        "blockers": list(blockers),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-smi", required=True, type=Path)
    parser.add_argument("--holdout-smi", type=Path)
    parser.add_argument("--prior", required=True, type=Path)
    parser.add_argument("--agent", required=True, type=Path)
    parser.add_argument("--staged-config", required=True, type=Path)
    parser.add_argument("--plugin-manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--out-manifest", required=True, type=Path)
    parser.add_argument("--command-template", required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--allow-empty-output", action="store_true")
    args = parser.parse_args()

    _remove_output(args.out_manifest)
    if args.seed < 0:
        raise SystemExit("--seed must be a non-negative integer")
    train_smiles = _read_smiles(args.train_smi, "REINVENT training SMILES")
    holdout_smiles = (
        _read_smiles(args.holdout_smi, "REINVENT holdout SMILES")
        if args.holdout_smi
        else []
    )
    overlap = sorted(set(train_smiles) & set(holdout_smiles))
    if overlap:
        raise SystemExit(
            "REINVENT training and holdout sets overlap after canonicalization: "
            + ", ".join(overlap[:10])
        )
    _require_nonempty(args.prior, "REINVENT prior artifact")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    values = {
        "train_smi": str(args.train_smi.resolve()),
        "prior": str(args.prior.resolve()),
        "out_dir": str(args.out_dir.resolve()),
        "agent": str(args.agent.resolve()),
        "staged_config": str(args.staged_config.resolve()),
        "plugin_manifest": str(args.plugin_manifest.resolve()),
        "seed": str(args.seed),
    }
    command = _format_command(args.command_template, values)
    environment = os.environ.copy()
    environment["REINVENT_SEED"] = str(args.seed)
    try:
        result = subprocess.run(
            command,
            cwd=args.out_dir,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        blockers = [f"fine-tune command could not start: {exc}"]
        if not args.allow_empty_output:
            raise SystemExit("; ".join(blockers)) from exc
        _write_json_atomic(args.out_manifest, _blocked_payload(args, blockers))
        return

    if result.returncode != 0:
        blockers = [
            f"fine-tune command exited with code {result.returncode}",
            result.stderr.strip()[-2000:] or "no stderr captured",
        ]
        if not args.allow_empty_output:
            raise SystemExit("; ".join(blockers))
        _write_json_atomic(args.out_manifest, _blocked_payload(args, blockers))
        return

    required_outputs = (
        (args.agent, "REINVENT agent artifact"),
        (args.staged_config, "REINVENT staged-learning config"),
        (args.plugin_manifest, "REINVENT plugin manifest"),
    )
    missing = [f"{label}: {path}" for path, label in required_outputs if not path.exists()]
    if missing:
        blockers = ["fine-tune completed without required output artifacts"] + missing
        if not args.allow_empty_output:
            raise SystemExit("; ".join(blockers))
        _write_json_atomic(args.out_manifest, _blocked_payload(args, blockers))
        return

    artifacts = [_artifact(args.prior, "prior")]
    artifacts.extend(_artifact(path, label) for path, label in required_outputs)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": "5.6",
        "component": "reinvent4_transfer_learning",
        "execution_status": "ready",
        "claim_eligible": True,
        "diagnostic_only": False,
        "seed": args.seed,
        "train_smiles_count": len(train_smiles),
        "holdout_smiles_count": len(holdout_smiles),
        "command": command,
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
        "artifacts": artifacts,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
    }
    _write_json_atomic(args.out_manifest, payload)


if __name__ == "__main__":
    main()
