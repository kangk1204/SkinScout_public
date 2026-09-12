#!/usr/bin/env python3
"""User-facing SkinScout launcher for one compound.

This wrapper keeps the operator-facing contract small: provide one SMILES (or
one SDF), choose the depth of analysis, and run the existing Snakemake workflow
with the same fail-closed rules used by CI.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
EVAL_DIR = ROOT / "eval"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from compound_applicability import (  # noqa: E402
    assess as assess_applicability,
    assess_sdf as assess_sdf_applicability,
    refusal_message as applicability_refusal,
)
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import stage0_verify  # noqa: E402
import gpu_admission  # noqa: E402
from discovery_canonical import discovery_key, discovery_key_from_mol  # noqa: E402
from leakage_check import (  # noqa: E402
    Thresholds as LeakageThresholds,
    inspect_manifest_bound_direct_reference,
    validate_discovery_alias_package,
    validate_sealed_discovery_audit,
    validate_sealed_discovery_audit_sources,
)
from data_readiness import (  # noqa: E402
    allow_stage0_build_message as _allow_stage0_build_message,
    check_required_artifact as _check_required_artifact,
    failed_entries_are_stage0_buildable as _failed_entries_are_stage0_buildable,
    failed_check_entry as _failed_check_entry,
    failure_message as _data_readiness_failure_message,
    required_data_artifacts as _required_data_artifacts,
    stage0_source_failures_for_build as _stage0_source_failures_for_build,
)
from skinscout.contracts.run_profile import (  # noqa: E402
    CONTEXT_PROFILES as SOTA_CONTEXT_PROFILES,
    EVIDENCE_MODES,
    MODES,
    PRESETS,
    RunProfile,
    build_run_manifest_v2,
    build_run_manifest_v3,
    canonical_json_sha256,
    migrate_run_manifest_v2,
    resolve_run_profile,
    validate_run_manifest,
)

RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
DEFAULT_STAGE0_SMILES = "C"
DEFAULT_RESULTS_ROOT = Path(
    os.environ.get("SKINSCOUT_RESULTS_ROOT") or "results/runs"
)
DEFAULT_TARGET_METADATA = Path("data/hpa/proteinatlas.tsv")
SAFETY_PRESETS = {"safety", "target-id", "report"}
SAFETY_DEGRADED_CONFIG = [
    "admet.allow_admet_ai_unavailable=true",
    "admet.allow_skin_sens_unavailable=true",
]
SOTA_CLAIM_CONFIG = [
    "evaluation.sota.enabled=true",
    "evaluation.sota.skin_known_min_case_top10=0.80",
    "evaluation.sota.skin_known_min_target_top10=0.50",
    "evaluation.sota.skin_known_min_target_top30=0.60",
    # Claim evaluation must not score the same known compound-target labels
    # that it later measures for recovery. Priors remain available for an
    # explicitly configured production/diagnostic run.
    "known_target_priors.enabled=false",
    "known_target_priors.weight=0.0",
    "known_target_priors.similarity_enabled=false",
]
IMMUTABLE_DISCOVERY_CONFIG = [
    "evidence_mode=discovery",
    "known_target_priors.enabled=false",
    "known_target_priors.weight=0.0",
    "known_target_priors.similarity_enabled=false",
    "docking.daina_evidence_mode=leave-query-out",
    # Discovery binds the exact ChEMBL snapshot it scored against - the metadata
    # names the fingerprint and activity tables and their sha256, checked by
    # _validate_discovery_daina_metadata. The recipe path scores a prebuilt index
    # and records the index instead, so the two contracts do not meet. Discovery
    # already pins evidence_mode for the same reason; pin this the same way
    # rather than making the mode unusable whenever the flag is on.
    "docking.daina_recipe_scoring=false",
]
SOTA_CONTEXT_PROFILE_OVERRIDES: dict[str, tuple[str, ...]] = {
    "general_skin": (),
    "pigmentation": (
        "skin_weight=0.42",
        "skin_min_threshold=0.06",
        "docking.top_pct_to_rescore=0.015",
    ),
    "anti_aging": (
        "skin_weight=0.35",
        "docking.min_sources_comprehensive=4",
    ),
    "barrier": (
        "skin_weight=0.40",
        "skin_min_threshold=0.06",
        "skin_expression.require_proteome_source=true",
        "skin_expression.require_gtex_source=true",
    ),
    "acne": (
        "skin_weight=0.44",
        "docking.top_pct_to_rescore=0.03",
    ),
    "inflammation": (
        "skin_weight=0.37",
        "skin_min_threshold=0.05",
    ),
    "irritation_sensitization": (
        "skin_weight=0.38",
        "skin_expression.require_gtex_source=true",
    ),
}
CONFIG_KEY_RE = re.compile(r"^[A-Za-z_][\w-]*\w$")
CONFIG_NUMERIC_RE = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"
)
LAUNCHER_CONFIG_KEYS = frozenset({
    "run_id", "mode", "compound_smiles", "compound_sdf", "evidence_mode",
    "run_dti_sanity", "target_metadata",
})
REPORT_CONDA_READINESS = {"bioemu_env", "md_env", "qm_env"}
REPORT_ACTIVE_READINESS = {
    "acpype",
    "bioemu",
    "crest",
    "gmx_mmpbsa",
    "gromacs",
    "mdtraj",
    "obabel",
    "pyscf",
    "sklearn",
    "xtb",
}
VERIFICATION_JSON = "run_verification.json"
VERIFICATION_LOG = "run_verification.log"
RUN_MANIFEST_JSON = "run_manifest.json"
VERIFIER_OUTPUT_SCHEMA = "skinscout.run_output_verification.v1"
DISCOVERY_AUDIT_SCHEMA = "skinscout.discovery-leakage-audit.v2"
COMPLETED_ADMET_METRICS = (
    "AMES",
    "ClinTox",
    "DILI",
    "Skin_Reaction",
    "hERG",
    "LD50_Zhu",
    "Solubility_AqSolDB",
    "logP",
    "QED",
    "tpsa",
)
BASE_COMPLETED_VERIFIED_ARTIFACTS = (
    ("run_summary_json", "run_summary.json"),
    ("run_summary_md", "run_summary.md"),
    ("run_verification_log", VERIFICATION_LOG),
)
REPORT_COMPLETED_VERIFIED_ARTIFACTS = (
    ("html_report", "09_report/index.html"),
)


def _die(message: str) -> None:
    raise SystemExit(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    _write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _file_fingerprint(path: Path, name: str) -> tuple[dict[str, object] | None, str | None]:
    try:
        if path.is_symlink():
            return None, f"{name} must not be a symlink: {path}"
        if not path.exists():
            return None, f"{name} is missing: {path}"
        if not path.is_file():
            return None, f"{name} is not a file: {path}"
        data = path.read_bytes()
    except OSError as exc:
        return None, f"{name} could not be read: {path}: {exc}"
    if not data:
        return None, f"{name} is empty: {path}"
    return {
        "name": name,
        "path": str(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }, None


def _completed_verified_artifact_snapshot(
    run_dir: Path,
    preset: str,
) -> tuple[list[dict[str, object]], str | None]:
    snapshot: list[dict[str, object]] = []
    try:
        specs = _completed_verified_artifact_specs(preset, run_dir)
    except ValueError as exc:
        return snapshot, str(exc)
    for name, filename in specs:
        artifact, error = _file_fingerprint(run_dir / filename, name)
        if error is not None:
            return snapshot, error
        if artifact is not None:
            snapshot.append(artifact)
    return snapshot, None


def _completed_verified_artifact_specs(
    preset: str, run_dir: Path | None = None,
) -> tuple[tuple[str, str], ...]:
    specs = BASE_COMPLETED_VERIFIED_ARTIFACTS
    if run_dir is not None and preset in {"target-id", "report"}:
        summary_path = run_dir / "run_summary.json"
        try:
            summary = json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"target ranking seal could not read run summary: {exc}"
            ) from exc
        artifacts = summary.get("artifacts") if isinstance(summary, dict) else None
        relative = artifacts.get("target_ranking") if isinstance(artifacts, dict) else None
        if (
            not isinstance(relative, str)
            or not relative.strip()
            or relative != relative.strip()
            or Path(relative).is_absolute()
        ):
            raise ValueError("target ranking seal requires a relative summary artifact path")
        ranking_path = run_dir / relative
        if not ranking_path.resolve().is_relative_to(run_dir.resolve()):
            raise ValueError("target ranking seal path escapes the run directory")
        specs += (("target_ranking", relative),)
    if preset == "report":
        specs += REPORT_COMPLETED_VERIFIED_ARTIFACTS
    if run_dir is not None and preset in {"target-id", "report"}:
        fast = run_dir / "03_targets/mode_fast"
        if (fast / "top50_band_reranked.csv").exists():
            specs += (
                ("target_fast_original_ranking", "03_targets/mode_fast/top50.csv"),
                ("target_fast_band_ranking", "03_targets/mode_fast/top50_band_reranked.csv"),
                ("target_fast_band_targets", "03_targets/mode_fast/daina_band_reranked_targets.csv"),
            )
    return specs


def _validate_completed_verified_artifacts(
    payload: dict[str, object],
    run_dir: Path,
    preset: str,
) -> None:
    try:
        specs = _completed_verified_artifact_specs(preset, run_dir)
    except ValueError as exc:
        _die(f"Completed run verification target ranking is invalid: {exc}")
    artifacts = payload.get("verified_artifacts")
    if (
        not isinstance(artifacts, list)
        or len(artifacts) != len(specs)
    ):
        _die("Completed run verification record verified_artifacts are incomplete")
    expected = {
        name: run_dir / filename
        for name, filename in specs
    }
    seen: set[str] = set()
    for idx, artifact in enumerate(artifacts, start=1):
        if not isinstance(artifact, dict):
            _die(f"Completed run verification artifact {idx} must be an object")
        name = artifact.get("name")
        if not isinstance(name, str) or name not in expected:
            _die(f"Completed run verification artifact {idx} has unexpected name")
        if name in seen:
            _die(f"Completed run verification artifact is duplicated: {name}")
        seen.add(name)
        expected_path = expected[name]
        if not expected_path.resolve().is_relative_to(run_dir.resolve()):
            _die(f"Completed run verification artifact escapes run directory: {name}")
        if artifact.get("path") != str(expected_path):
            _die(f"Completed run verification artifact path does not match: {name}")
        current, error = _file_fingerprint(expected_path, name)
        if error is not None:
            _die(f"Completed run verification artifact is invalid: {error}")
        if current is None:
            _die(f"Completed run verification artifact is invalid: {name}")
        if artifact.get("bytes") != current["bytes"]:
            _die(f"Completed run verification artifact bytes changed: {name}")
        if artifact.get("sha256") != current["sha256"]:
            _die(f"Completed run verification artifact sha256 changed: {name}")
    missing = sorted(set(expected) - seen)
    if missing:
        _die(
            "Completed run verification record missing verified artifact(s): "
            + ", ".join(missing)
        )


def _validate_completed_summary_identity(
    args: argparse.Namespace,
    run_dir: Path,
    payload: dict[str, object],
) -> None:
    if payload.get("schema_version") != "skinscout.run_summary.v1":
        _die("Completed run summary has invalid schema_version")
    if payload.get("run_id") != run_dir.name:
        _die("Completed run summary run_id does not match this run")
    if payload.get("preset") != args.preset:
        _die("Completed run summary preset does not match this run")
    if payload.get("mode") != args.mode:
        _die("Completed run summary mode does not match this run")


def _canonical_smiles(smiles: str) -> str:
    text = smiles.strip()
    if not text:
        _die("--smiles must be non-empty")
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        _die(f"Could not parse SMILES: {smiles!r}")
    return Chem.MolToSmiles(mol, canonical=True)


def _require_in_scope(result: dict[str, object]) -> None:
    """Refuse an input the documented scope excludes, before building the DAG.

    The gate used to live only in the Workbench SMILES field, so the CLI and
    every SDF upload walked straight past it.
    """
    message = applicability_refusal(result)
    if message is not None:
        _die(message)


def _validate_sdf(path: Path) -> Path:
    if not path.exists() or path.stat().st_size == 0:
        _die(f"--sdf must point to a non-empty SDF file: {path}")
    return path


def _validate_run_id(run_id: str) -> str:
    text = run_id.strip()
    if text in {"", ".", ".."} or not RUN_ID_RE.fullmatch(text):
        _die(
            "run_id must use only letters, numbers, '.', '_' and '-' and must "
            f"start with a letter or number: {run_id!r}"
        )
    return text


def _auto_run_id_from_smiles(smiles: str) -> str:
    canonical = _canonical_smiles(smiles)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    return _validate_run_id(f"smiles_{digest}")


def _auto_run_id_from_sdf(path: Path) -> str:
    sdf = _validate_sdf(path)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", sdf.stem).strip("._-")
    if not stem:
        stem = "compound"
    return _validate_run_id(f"sdf_{stem[:124]}")


def _resolve_run_id(args: argparse.Namespace) -> str:
    if args.run_id:
        return _validate_run_id(args.run_id)
    if args.preset == "stage0" and not args.smiles and not args.sdf:
        return "stage0_bootstrap"
    if args.smiles and not args.sdf:
        return _auto_run_id_from_smiles(args.smiles)
    if args.sdf and not args.smiles:
        return _auto_run_id_from_sdf(args.sdf)
    _die("run_id could not be inferred; provide exactly one of --smiles or --sdf")


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _normalize_preset_args(args: argparse.Namespace) -> None:
    try:
        profile = resolve_run_profile(
            args.preset,
            args.mode,
            evidence_mode=getattr(args, "evidence_mode", "evidence"),
            context_profile=getattr(args, "context_profile", "auto"),
            sota_claim=bool(getattr(args, "sota_claim", False)),
        )
    except ValueError as exc:
        _die(str(exc))
    args.run_profile = profile
    for key, value in profile.legacy_fields().items():
        setattr(args, key, value)
    args.evidence_mode = profile.evidence_mode.value


def _sota_config_entries(args: argparse.Namespace) -> list[str]:
    entries: list[str] = []
    context_profile = getattr(args, "context_profile", "auto")
    if args.sota_claim:
        entries.extend(SOTA_CLAIM_CONFIG)
    if args.sota_claim or context_profile != "auto":
        entries.append(f"evaluation.sota.default_context_profile={context_profile}")
        entries.extend(SOTA_CONTEXT_PROFILE_OVERRIDES.get(context_profile, ()))
    return entries


def _preset_targets(preset: str, mode: str, run_dti_sanity: bool) -> list[str]:
    if preset == "stage0":
        return ["stage0_complete", "activity_retrieval_operational_gate"]
    if preset == "safety":
        return ["skin_sens_consensus", "cosmetic_drug_decision"]
    if preset == "target-id":
        targets = ["kg_efficacy_label"]
        if mode in {"comprehensive", "both"} and run_dti_sanity:
            targets.append("disagreement_analysis")
        if mode == "both":
            targets.append("fast_rerank_consensus")
        return targets
    if preset == "report":
        return ["all"]
    raise AssertionError(f"Unhandled preset: {preset}")


def _required_models(preset: str, mode: str, run_dti_sanity: bool) -> list[str]:
    if preset in {"stage0", "safety"}:
        return []
    required = {"gpu", "gnina"}
    if mode in {"comprehensive", "both"}:
        # autodock_pick_top_pct takes diffdock_blind_no_pocket's scores as a
        # hard input, so a machine without DiffDock reported ready and then
        # died mid-DAG.
        required.update({"boltz", "rtmscore", "diffdock"})
        if run_dti_sanity:
            required.add("psichic")
    return sorted(required)


def _required_model_readiness(
    preset: str,
    mode: str,
    run_dti_sanity: bool,
    *,
    use_conda: bool,
) -> list[str]:
    required = set(_required_models(preset, mode, run_dti_sanity))
    if preset in {"target-id", "report"}:
        required.update({"autodock_gpu", "autogrid"})
        if use_conda:
            required.update({"autodock_gpu_env", "meeko_env"})
        else:
            required.add("meeko")
    if preset == "report":
        if use_conda:
            required.update(REPORT_CONDA_READINESS)
        else:
            required.update(REPORT_ACTIVE_READINESS)
    return sorted(required)


def _run_data_readiness(
    preset: str,
    mode: str,
    run_dti_sanity: bool,
    *,
    allow_stage0_build: bool,
) -> None:
    missing = [
        _failed_check_entry(check)
        for label, path in _required_data_artifacts(preset, mode)
        if (check := _check_required_artifact(label, path)).status != "present"
    ]
    if not missing:
        return
    if allow_stage0_build and _failed_entries_are_stage0_buildable(missing):
        source_missing = _stage0_source_failures_for_build()
        if source_missing:
            _die(_data_readiness_failure_message(source_missing))
        sys.stderr.write(_allow_stage0_build_message(missing) + "\n")
        return
    _die(_data_readiness_failure_message(missing))


def _readiness_base_command(
    readiness_command: str | None,
    requirements: list[str],
    *,
    use_conda: bool = True,
) -> list[str]:
    if not use_conda:
        if readiness_command and shlex.split(readiness_command) != [sys.executable]:
            _die("--no-use-conda requires readiness in the current Python environment; omit --readiness-command")
        return [sys.executable]
    if readiness_command:
        return shlex.split(readiness_command)
    if {"boltz", "rtmscore", "psichic"}.intersection(requirements) and shutil.which(
        "micromamba"
    ):
        return ["micromamba", "run", "-n", "cosmax-boltz2", "python"]
    return [sys.executable]


def _resolve_conda_frontend(frontend: str, *, validate: bool) -> str | None:
    if frontend != "auto":
        available = shutil.which(frontend)
        if frontend == "mamba" and available is None:
            available = shutil.which("micromamba")
        if validate and available is None:
            _die(
                f"Snakemake conda frontend '{frontend}' is not on PATH; "
                "install it or rerun with --no-use-conda"
            )
        return frontend
    for candidate in ("conda", "mamba"):
        if shutil.which(candidate):
            return candidate
    if shutil.which("micromamba"):
        return "mamba"
    if validate:
        _die(
            "Snakemake --use-conda requires conda, mamba, or micromamba on PATH. "
            "Install one of them, or rerun with --no-use-conda only when the "
            "active environment already contains every rule dependency."
        )
    return None


def _snakemake_environment(
    args: argparse.Namespace,
) -> tuple[dict[str, str], tempfile.TemporaryDirectory[str] | None]:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    snakemake_command = getattr(args, "snakemake", "snakemake")
    if (
        Path(snakemake_command).name == snakemake_command
        and shutil.which(snakemake_command) is None
    ):
        runtime_command = Path(sys.executable).resolve().with_name(snakemake_command)
        if runtime_command.is_file() and os.access(runtime_command, os.X_OK):
            env["PATH"] = str(runtime_command.parent) + os.pathsep + env.get("PATH", "")
    if (
        not args.use_conda
        or shutil.which("mamba") is not None
        or _resolve_conda_frontend(args.conda_frontend, validate=False) != "mamba"
    ):
        return env, None
    micromamba = shutil.which("micromamba")
    if micromamba is None:
        return env, None
    shim_dir = tempfile.TemporaryDirectory(prefix="skinscout-mamba-")
    Path(shim_dir.name, "mamba").symlink_to(Path(micromamba).resolve())
    env["PATH"] = shim_dir.name + os.pathsep + env.get("PATH", "")
    return env, shim_dir


def _run_model_readiness(
    requirements: list[str], readiness_command: str | None, *, use_conda: bool = True
) -> None:
    if not requirements:
        return
    cmd = _readiness_base_command(readiness_command, requirements, use_conda=use_conda)
    cmd.append("scripts/model_readiness.py")
    if not use_conda:
        cmd.append("--active-environment-only")
    for requirement in requirements:
        cmd.extend(["--require", requirement])
    res = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    if res.returncode != 0:
        sys.stderr.write(res.stdout)
        sys.stderr.write(res.stderr)
        _die("Model readiness preflight failed")


def _requires_gpu_admission(preset: str, *, dry_run: bool) -> bool:
    return not dry_run and preset in {"target-id", "report"}


def _run_gpu_admission() -> None:
    status = gpu_admission.probe(cwd=ROOT)
    if status.get("available_for_analysis") is True:
        return
    detail = str(status.get("detail") or "GPU 상태를 확인하지 못했습니다.")
    _die(
        "GPU execution admission failed: "
        f"{detail} 다른 GPU 계산이 끝난 뒤 다시 실행하세요. "
        "이 실행 전 게이트는 readiness skip 옵션으로 우회할 수 없습니다."
    )


def _safety_readiness_base_command(readiness_command: str | None) -> list[str]:
    if readiness_command:
        return shlex.split(readiness_command)
    return [sys.executable]


def _run_safety_readiness(
    *,
    use_conda: bool,
    readiness_command: str | None,
    online: bool,
    allow_degraded: bool,
) -> None:
    cmd = _safety_readiness_base_command(readiness_command)
    cmd.extend(
        [
            "scripts/safety_readiness.py",
            "--runtime-mode",
            "conda" if use_conda else "active",
        ]
    )
    if allow_degraded:
        cmd.append("--allow-degraded")
    res = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    if res.returncode != 0:
        sys.stderr.write(res.stdout)
        sys.stderr.write(res.stderr)
        _die("Safety readiness preflight failed")

    if not online:
        return
    online_cmd = [*cmd, "--online"]
    online_res = subprocess.run(
        online_cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if online_res.returncode == 0:
        return
    sys.stderr.write(online_res.stdout)
    sys.stderr.write(online_res.stderr)
    if allow_degraded:
        sys.stderr.write(
            "Safety web-service readiness failed; continuing because "
            "--allow-safety-degraded was set\n"
        )
        return
    _die(
        "Safety web-service readiness failed; rerun with --allow-safety-degraded "
        "only for an explicit degraded run"
    )


def _run_stage0_claim_quality_readiness() -> None:
    checks = stage0_verify.collect_checks(
        ROOT,
        claim_quality=True,
        check_stage0_flag=True,
    )
    failed = [check for check in checks if not check.ok]
    if not failed:
        return
    sys.stderr.write("Stage 0 claim-quality preflight failed:\n")
    for check in failed:
        sys.stderr.write(f"  - {check.name}: {check.detail}\n")
    sys.stderr.write("Run: python scripts/stage0_verify.py --strict --claim-quality\n")
    _die("Stage 0 claim-quality preflight failed")


def _validate_extra_config(values: list[str]) -> list[str]:
    for value in values:
        if "=" not in value or value.startswith("="):
            _die(f"--extra-config values must use key=value syntax: {value!r}")
    return values


def _validate_user_config(values: list[str]) -> None:
    """Keep the launcher identity and the workflow identity identical."""
    seen: set[str] = set()
    for value in _validate_extra_config(values):
        key, _, raw = value.partition("=")
        if key.split(".", 1)[0] in LAUNCHER_CONFIG_KEYS:
            _die(f"--extra-config cannot override launcher-owned key: {key}")
        if key == "paths":
            _die("--extra-config paths must use dotted keys such as paths.results_root")
        if any(key == old or key.startswith(old + ".") or old.startswith(key + ".") for old in seen):
            _die(f"--extra-config contains duplicate or conflicting key: {key}")
        if key == "paths.results_root" and not raw.strip():
            _die("--extra-config paths.results_root must be non-empty")
        seen.add(key)


def _validate_config_key(key: str, raw: str) -> None:
    if not CONFIG_KEY_RE.fullmatch(key):
        _die(f"--extra-config contains invalid config key {key!r}: {raw!r}")


def _nested_config_value(raw: str) -> str:
    value = raw.strip()
    if CONFIG_NUMERIC_RE.fullmatch(value):
        return value
    if value.lower() in {"true", "false", "null", "none"}:
        return value.lower()
    return repr(value)


def _insert_nested_config(root: dict[str, object], keys: list[str], value: str) -> None:
    cursor = root
    for key in keys[:-1]:
        child = cursor.setdefault(key, {})
        if not isinstance(child, dict):
            _die(f"--extra-config conflicts with nested config key: {'.'.join(keys)}")
        cursor = child
    cursor[keys[-1]] = value


def _format_nested_config(value: object) -> str:
    if isinstance(value, dict):
        return "{" + ", ".join(
            f"{key!r}: {_format_nested_config(child)}"
            for key, child in sorted(value.items())
        ) + "}"
    if isinstance(value, str):
        return _nested_config_value(value)
    raise AssertionError(f"Unexpected config value: {value!r}")


def _snakemake_config_entries(values: list[str]) -> list[str]:
    flat: list[str] = []
    nested: dict[str, object] = {}
    for value in _validate_extra_config(values):
        key, _, raw = value.partition("=")
        parts = key.split(".")
        if any(part == "" for part in parts):
            _die(f"--extra-config contains an empty config key segment: {value!r}")
        for part in parts:
            _validate_config_key(part, value)
        if len(parts) == 1:
            flat.append(value)
            continue
        _insert_nested_config(nested, parts, raw)
    return [
        *flat,
        *[
            f"{key}={_format_nested_config(value)}"
            for key, value in sorted(nested.items())
        ],
    ]


def _skipped_readiness_checks(args: argparse.Namespace) -> list[tuple[str, str]]:
    return [
        (label, flag)
        for label, flag, enabled in (
            ("data readiness", "--skip-data-readiness", args.skip_data_readiness),
            (
                "safety readiness",
                "--skip-safety-readiness",
                args.skip_safety_readiness,
            ),
            ("model readiness", "--skip-model-readiness", args.skip_model_readiness),
        )
        if enabled
    ]


def _validate_readiness_skip_policy(args: argparse.Namespace) -> None:
    skipped = [flag for _label, flag in _skipped_readiness_checks(args)]
    if not skipped:
        return
    if args.dry_run or args.allow_unsafe_readiness_skip:
        return
    _die(
        "Readiness preflight skips are only allowed for --dry-run. "
        "For an actual workflow execution, remove "
        f"{', '.join(skipped)} or add --allow-unsafe-readiness-skip for an "
        "explicit diagnostic run that must not be treated as a prediction claim."
    )


def _validate_output_verification_skip_policy(args: argparse.Namespace) -> None:
    if not args.skip_output_verification:
        return
    if args.dry_run or args.allow_unsafe_readiness_skip:
        return
    _die(
        "Completed-run output verification cannot be skipped for an actual "
        "workflow execution. Remove --skip-output-verification or add "
        "--allow-unsafe-readiness-skip for an explicit diagnostic run that must "
        "not be treated as a prediction claim."
    )


DISCOVERY_AUDIT_SOURCE_FILENAMES = {
    "input_csv": "eval_targets.csv",
    "training_seq_db": "training_cutoff_seqs.fasta",
    "training_ligands": "training_ligands.smi",
    "training_holo": "training_holo_pockets.csv",
    "direct_exact_reference": "direct_exact_reference.smi",
    "leakage_audit_csv": "leakage_audit.csv",
}
DISCOVERY_AUDIT_THRESHOLDS = LeakageThresholds(0.30, 0.50, 0.50)
DISCOVERY_AUDIT_ACCEPTANCE_FILENAME = "discovery_audit_acceptance.json"
DISCOVERY_DIRECT_PREFLIGHT_FILENAME = "discovery_direct_preflight.json"


def _active_discovery_audit_sources(run_dir: Path) -> dict[str, Path]:
    source_dir = run_dir / "discovery_audit_sources"
    return {
        key: source_dir / filename
        for key, filename in DISCOVERY_AUDIT_SOURCE_FILENAMES.items()
    }


def _discovery_acceptance_binding(payload: dict[str, object]) -> str:
    unsigned = dict(payload)
    unsigned.pop("binding_sha256", None)
    canonical = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _discovery_alias_dir(args: argparse.Namespace) -> Path:
    path = args.discovery_alias_dir
    return path if path.is_absolute() else ROOT / path


def _validate_discovery_alias_inputs(
    args: argparse.Namespace,
) -> tuple[Path, Path, dict[str, object]]:
    alias_dir = _discovery_alias_dir(args)
    reference = alias_dir / "direct_exact_reference.smi"
    manifest = alias_dir / "manifest.json"
    canonical_alias_dir = (ROOT / "data" / "discovery_aliases").resolve()
    if alias_dir.resolve() == canonical_alias_dir:
        check = stage0_verify.chk_discovery_alias_integrity(ROOT)
        if not check.ok:
            raise ValueError(
                "canonical Stage 0 Discovery alias integrity failed: "
                f"{check.detail}"
            )
    payload = validate_discovery_alias_package(reference, manifest)
    return reference, manifest, payload


def _discovery_input_identity(args: argparse.Namespace) -> tuple[str, str]:
    if args.smiles:
        try:
            identity = discovery_key(args.smiles, label="Discovery input SMILES")
        except ValueError as exc:
            _die(str(exc))
        return identity.discovery_key_sha256, identity.parent_canonical_smiles
    if args.sdf:
        sdf = _validate_sdf(args.sdf)
        try:
            supplier = Chem.SDMolSupplier(str(sdf), removeHs=False)
            mol = supplier[0] if len(supplier) else None
        except Exception as exc:
            _die(f"Could not parse SDF for Discovery audit binding: {sdf}: {exc}")
        if mol is None:
            _die(f"Could not parse SDF for Discovery audit binding: {sdf}")
        try:
            identity = discovery_key_from_mol(
                mol,
                input_text=str(sdf),
                label="Discovery input SDF",
            )
        except ValueError as exc:
            _die(str(exc))
        return identity.discovery_key_sha256, identity.parent_canonical_smiles
    _die("Discovery mode requires exactly one of --smiles or --sdf")


def _discovery_direct_preflight_binding(payload: dict[str, object]) -> str:
    unsigned = dict(payload)
    unsigned.pop("binding_sha256", None)
    return canonical_json_sha256(unsigned)


def _fresh_discovery_direct_preflight(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    expected_input_key: str,
    parent_canonical_smiles: str,
) -> None:
    try:
        reference, manifest, alias_manifest = _validate_discovery_alias_inputs(args)
        inspection = inspect_manifest_bound_direct_reference(
            reference,
            parent_canonical_smiles=parent_canonical_smiles,
            discovery_key_sha256=expected_input_key,
            payload=alias_manifest,
        )
    except (OSError, ValueError) as exc:
        _die(f"Discovery direct-reference preflight failed: {exc}")
    if inspection["matched"]:
        _die(
            "Discovery mode rejected an exact compound already present in the "
            "direct-evidence reference. Use Evidence mode for known compounds."
        )
    reference_record = {
        "name": "discovery direct reference",
        "path": inspection["path"],
        "bytes": inspection["bytes"],
        "sha256": inspection["sha256"],
    }
    manifest_record, manifest_error = _file_fingerprint(
        manifest,
        "discovery alias manifest",
    )
    if (
        manifest_error
        or manifest_record is None
    ):
        _die(manifest_error or "Discovery preflight binding failed")
    payload: dict[str, object] = {
        "schema_version": "skinscout.discovery-direct-preflight.v1",
        "run_id": run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "input_discovery_key_sha256": expected_input_key,
        "parent_canonical_smiles": parent_canonical_smiles,
        "direct_exact_match": False,
        "reference": reference_record,
        "manifest": manifest_record,
        "policy": (
            "exclusion-only preflight; direct records are not workflow inputs, "
            "features, candidate generators, or ranking support"
        ),
        "accepted_at_utc": _utc_now(),
    }
    payload["binding_sha256"] = _discovery_direct_preflight_binding(payload)
    _write_json_atomic(run_dir / DISCOVERY_DIRECT_PREFLIGHT_FILENAME, payload)


def _validate_existing_discovery_direct_preflight(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    expected_input_key: str,
    parent_canonical_smiles: str,
) -> None:
    path = run_dir / DISCOVERY_DIRECT_PREFLIGHT_FILENAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Discovery direct preflight is unreadable: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Discovery direct preflight must be a JSON object")
    if payload.get("schema_version") != "skinscout.discovery-direct-preflight.v1":
        raise ValueError("Discovery direct preflight schema mismatch")
    if payload.get("binding_sha256") != _discovery_direct_preflight_binding(payload):
        raise ValueError("Discovery direct preflight binding mismatch")
    expected = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "input_discovery_key_sha256": expected_input_key,
        "parent_canonical_smiles": parent_canonical_smiles,
        "direct_exact_match": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Discovery direct preflight {key} mismatch")
    reference, manifest, alias_manifest = _validate_discovery_alias_inputs(args)
    inspection = inspect_manifest_bound_direct_reference(
        reference,
        parent_canonical_smiles=parent_canonical_smiles,
        discovery_key_sha256=expected_input_key,
        payload=alias_manifest,
    )
    active_reference = {
        "name": "discovery direct reference",
        "path": inspection["path"],
        "bytes": inspection["bytes"],
        "sha256": inspection["sha256"],
    }
    if payload.get("reference") != active_reference:
        raise ValueError("Discovery direct preflight reference binding mismatch")
    active_manifest, manifest_error = _file_fingerprint(
        manifest,
        "discovery alias manifest",
    )
    if manifest_error is not None or active_manifest is None or payload.get(
        "manifest"
    ) != active_manifest:
        raise ValueError("Discovery direct preflight manifest binding mismatch")
    if inspection["matched"]:
        raise ValueError("Discovery direct preflight input is now an exact match")


def _validate_existing_discovery_acceptance(
    *,
    run_dir: Path,
    audit_path: Path,
    payload: dict[str, object],
    expected_input_key: str,
) -> None:
    path = run_dir / DISCOVERY_AUDIT_ACCEPTANCE_FILENAME
    try:
        acceptance = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Discovery audit acceptance is unreadable: {path}: {exc}") from exc
    if not isinstance(acceptance, dict):
        raise ValueError("Discovery audit acceptance must be a JSON object")
    if acceptance.get("schema_version") != "skinscout.discovery-audit-acceptance.v1":
        raise ValueError("Discovery audit acceptance schema mismatch")
    if acceptance.get("binding_sha256") != _discovery_acceptance_binding(acceptance):
        raise ValueError("Discovery audit acceptance binding mismatch")
    expected = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "input_discovery_key_sha256": expected_input_key,
        "audit_path": str(audit_path.resolve()),
        "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "audit_binding_sha256": payload.get("binding_sha256"),
        "execution_challenge": payload.get("execution_challenge"),
    }
    for key, value in expected.items():
        if acceptance.get(key) != value:
            raise ValueError(f"Discovery audit acceptance {key} mismatch")
    validate_sealed_discovery_audit(payload, expected_input_key=expected_input_key)
    validate_sealed_discovery_audit_sources(
        payload,
        expected_sources=_active_discovery_audit_sources(run_dir),
        expected_thresholds=DISCOVERY_AUDIT_THRESHOLDS,
        expected_execution_challenge=str(payload["execution_challenge"]),
        require_direct_exact_reference=True,
    )


def _validate_evidence_mode_execution(args: argparse.Namespace) -> None:
    if args.evidence_mode == "evidence":
        return
    run_dir = _run_dir(args)
    preflight_path = run_dir / DISCOVERY_DIRECT_PREFLIGHT_FILENAME
    audit_path = run_dir / "discovery_leakage_audit.json"
    expected_key, parent_canonical_smiles = _discovery_input_identity(args)
    if not args.verify_existing_run:
        try:
            _fresh_discovery_direct_preflight(
                args=args,
                run_dir=run_dir,
                expected_input_key=expected_key,
                parent_canonical_smiles=parent_canonical_smiles,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            _die(f"Discovery direct-reference preflight failed: {exc}")
        return
    if preflight_path.exists():
        try:
            _validate_existing_discovery_direct_preflight(
                args=args,
                run_dir=run_dir,
                expected_input_key=expected_key,
                parent_canonical_smiles=parent_canonical_smiles,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            _die(f"Discovery direct-reference preflight failed: {exc}")
        return
    if not audit_path.exists():
        _die(
            "Existing Discovery run has neither a direct preflight nor a "
            f"legacy leakage audit: {run_dir}"
        )
    if not audit_path.is_file() or audit_path.stat().st_size == 0:
        _die(f"Discovery leakage audit is empty or unsafe: {audit_path}")
    try:
        payload = json.loads(audit_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        _die(f"Discovery leakage audit is unreadable: {audit_path}: {exc}")
    if not isinstance(payload, dict):
        _die(f"Discovery leakage audit must be a JSON object: {audit_path}")
    try:
        validate_sealed_discovery_audit(
            payload,
            expected_input_key=expected_key,
        )
        _validate_existing_discovery_acceptance(
            run_dir=run_dir,
            audit_path=audit_path,
            payload=payload,
            expected_input_key=expected_key,
        )
    except (RuntimeError, ValueError) as exc:
        _die(f"Discovery leakage audit schema validation failed: {audit_path}: {exc}")
    if payload.get("status") != "ok" or payload.get("direct_exact_count") != 0:
        _die(
            "Discovery leakage audit must have status=ok and "
            f"direct_exact_count=0: {audit_path}"
        )
    counts = payload["counts"]
    if not isinstance(counts, dict) or int(counts.get("survivor_rows", 0)) < 1:
        _die(f"Discovery leakage audit has no survivor rows: {audit_path}")


def _results_root_from_extra_config(values: list[str]) -> Path:
    for value in values:
        key, sep, raw = value.partition("=")
        if sep and key == "paths.results_root" and raw.strip():
            return Path(raw.strip())
    return DEFAULT_RESULTS_ROOT


def _run_dir(args: argparse.Namespace) -> Path:
    root = _results_root_from_extra_config(args.extra_config)
    if not root.is_absolute():
        root = ROOT / root
    return root / _resolve_run_id(args)


def _profile_from_args(args: argparse.Namespace) -> RunProfile:
    profile = getattr(args, "run_profile", None)
    if isinstance(profile, RunProfile):
        return profile
    try:
        return resolve_run_profile(
            getattr(args, "requested_preset", args.preset),
            args.mode,
            evidence_mode=getattr(args, "evidence_mode", "evidence"),
            context_profile=getattr(args, "context_profile", "auto"),
            sota_claim=bool(getattr(args, "sota_claim", False)),
        )
    except ValueError as exc:
        _die(str(exc))


def _data_manifest_sha256() -> str | None:
    data_root = ROOT / "data"
    paths = sorted(
        path
        for path in data_root.rglob("*.json")
        if path.is_file() and "manifest" in path.name.lower()
    )
    if not paths:
        return None
    entries = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in paths
    ]
    return canonical_json_sha256(entries)


def _run_manifest_payload(
    args: argparse.Namespace,
    command: list[str] | None = None,
) -> dict[str, object]:
    if command is None:
        command = build_snakemake_command(
            args,
            validate_conda_frontend=False,
        )
    try:
        config_index = command.index("--config")
    except ValueError:
        _die("Snakemake command is missing --config for run manifest")
    profile = _profile_from_args(args)
    manifest_config = list(command[config_index + 1 :])
    if args.sdf is not None:
        sdf = _validate_sdf(args.sdf)
        sdf_sha256 = hashlib.sha256(sdf.read_bytes()).hexdigest()
        manifest_config = [
            (
                f"compound_sdf_sha256={sdf_sha256}"
                if entry.startswith("compound_sdf=")
                else entry
            )
            for entry in manifest_config
        ]
    config_payload = {
        # Bind defaults as well as overrides: a changed threshold in the YAML
        # must not be accepted as the same run simply because its CLI is equal.
        "workflow_config_sha256": hashlib.sha256(
            (ROOT / "workflow" / "config.yaml").read_bytes()
        ).hexdigest(),
        "run_profile": profile.to_dict(),
        "targets": _preset_targets(
            args.preset,
            args.mode,
            not args.no_dti_sanity,
        ),
        "snakemake_config": manifest_config,
    }
    builder = build_run_manifest_v3 if profile.expected_fast_artifact_ids() else build_run_manifest_v2
    return builder(
        run_id=_resolve_run_id(args),
        profile=profile,
        config_payload=config_payload,
        data_sha256=_data_manifest_sha256(),
    )


def _finalize_run_manifest(
    args: argparse.Namespace,
    command: list[str] | None = None,
) -> None:
    payload = _run_manifest_payload(args, command)
    if payload.get("schema_version") != "skinscout.run_manifest.v3":
        return
    run_dir = _run_dir(args)
    canonical = run_dir / "03_targets" / "mode_fast" / "daina_structural_targets.csv"
    top50 = run_dir / "03_targets" / "mode_fast" / "top50.csv"
    for label, path in (("canonical target-fast artifact", canonical), ("top50 compatibility projection", top50)):
        if not path.exists() or not path.is_file() or path.stat().st_size == 0:
            _die(f"Cannot finalize v3 run manifest; {label} is missing/empty: {path}")
    target_fast = payload.get("target_fast")
    if not isinstance(target_fast, dict):
        _die("Cannot finalize v3 run manifest; target_fast contract is invalid")
    target_fast["sha256"] = hashlib.sha256(canonical.read_bytes()).hexdigest()
    projection = target_fast.get("compatibility_projection")
    if not isinstance(projection, dict):
        _die("Cannot finalize v3 run manifest; compatibility projection is invalid")
    projection["sha256"] = hashlib.sha256(top50.read_bytes()).hexdigest()
    try:
        validate_run_manifest(payload)
    except ValueError as exc:
        _die(f"Cannot finalize v3 run manifest: {exc}")
    _write_json_atomic(run_dir / RUN_MANIFEST_JSON, payload)


def _ensure_run_manifest(
    args: argparse.Namespace,
    command: list[str] | None = None,
    *,
    legacy_summary: dict[str, object] | None = None,
) -> Path:
    run_dir = _run_dir(args)
    path = run_dir / RUN_MANIFEST_JSON
    if legacy_summary is None:
        expected = _run_manifest_payload(args, command)
    else:
        legacy_payload = dict(legacy_summary)
        legacy_payload.update(_profile_from_args(args).legacy_fields())
        legacy_payload["evidence_mode"] = args.evidence_mode
        expected = migrate_run_manifest_v2(legacy_payload)
    if path.exists():
        try:
            observed = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            _die(f"Existing run manifest could not be read: {path}: {exc}")
        if not isinstance(observed, dict):
            _die(f"Existing run manifest must be a JSON object: {path}")
        try:
            validate_run_manifest(observed)
        except ValueError as exc:
            _die(f"Existing run manifest is invalid: {exc}")
        for key in ("schema_version", "run_id", "run_profile"):
            if observed.get(key) != expected.get(key):
                _die(f"Existing run manifest {key} does not match this run")
        observed_hashes = observed.get("hashes")
        expected_hashes = expected.get("hashes")
        if not isinstance(observed_hashes, dict) or not isinstance(expected_hashes, dict):
            _die("Existing run manifest hashes are invalid")
        if legacy_summary is None:
            hash_keys = (
                "schema_sha256",
                "config_sha256",
                "image_sha256",
                "data_sha256",
            )
            mismatched = [
                key
                for key in hash_keys
                if observed_hashes.get(key) != expected_hashes.get(key)
            ]
        elif (
            observed_hashes.get("data_sha256") is None
            and observed_hashes.get("image_sha256") is None
        ):
            mismatched = [
                key
                for key in expected_hashes
                if observed_hashes.get(key) != expected_hashes.get(key)
            ]
        else:
            current = _run_manifest_payload(args, command)
            current_hashes = current.get("hashes")
            if not isinstance(current_hashes, dict):
                _die("Current run manifest hashes are invalid")
            mismatched = (
                ["config_sha256"]
                if observed_hashes.get("config_sha256")
                != current_hashes.get("config_sha256")
                else []
            )
        if mismatched:
            _die(
                "Existing run manifest hash mismatch: "
                + ", ".join(mismatched)
            )
        return path
    _write_json_atomic(path, expected)
    return path


def _verification_payload_path(run_dir: Path) -> Path:
    return run_dir / ".run_verification_payload.json"


def _verification_command(args: argparse.Namespace, run_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/verify_run_outputs.py",
        "--run-dir",
        str(run_dir),
        "--preset",
        args.preset,
        "--mode",
        args.mode,
        "--target-metadata",
        str(args.target_metadata),
        "--json-out",
        str(_verification_payload_path(run_dir)),
    ]
    if args.allow_safety_degraded:
        cmd.append("--allow-degraded")
    return cmd


def _verification_checks_contract(
    checks: object,
    run_dir: Path,
    *,
    subject: str,
) -> tuple[list[dict[str, object]], str | None]:
    if not isinstance(checks, list) or not checks:
        return [], f"{subject} checks must be a non-empty list"
    run_root = run_dir.resolve(strict=False)
    validated: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for idx, check in enumerate(checks, start=1):
        if not isinstance(check, dict):
            return [], f"{subject} check {idx} must be an object"
        name = check.get("name")
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            return [], f"{subject} check {idx} name must be a non-empty string"
        if any(token in name for token in ("\r", "\n", "\x00")):
            return [], f"{subject} check name is invalid: {idx}"
        if check.get("status") != "ok":
            return [], f"{subject} check is not ok: {name}"
        path_text = check.get("path")
        if (
            not isinstance(path_text, str)
            or not path_text.strip()
            or path_text != path_text.strip()
        ):
            return [], f"{subject} check {name} path must be a non-empty string"
        if any(token in path_text for token in ("\r", "\n", "\x00")):
            return [], f"{subject} check path is invalid: {name}"
        detail = check.get("detail")
        if not isinstance(detail, str):
            return [], f"{subject} check {name} detail must be a string"
        if any(token in detail for token in ("\r", "\n", "\x00")):
            return [], f"{subject} check detail is invalid: {name}"
        check_key = (name, path_text)
        if check_key in seen:
            return [], f"{subject} check is duplicated: {name}"
        seen.add(check_key)
        check_path = Path(path_text)
        if not check_path.is_absolute():
            return [], f"{subject} check path is relative: {name}"
        try:
            check_path.resolve(strict=False).relative_to(run_root)
        except ValueError:
            return [], f"{subject} check path is outside this run: {name}"
        if not check_path.exists() or not check_path.is_file():
            return [], f"{subject} check path is missing or not a file: {name}"
        validated.append(check)
    return validated, None


def _verifier_payload_contract_error(
    args: argparse.Namespace,
    run_dir: Path,
    verifier_payload: dict[str, object],
) -> str | None:
    if verifier_payload.get("schema_version") != VERIFIER_OUTPUT_SCHEMA:
        return "verifier JSON has invalid schema_version"
    verifier_status = verifier_payload.get("status")
    if verifier_status != "ok":
        return f"verifier JSON status was not ok: {verifier_status}"
    if verifier_payload.get("run_dir") != str(run_dir):
        return "verifier JSON run_dir does not match this run"
    if verifier_payload.get("preset") != args.preset:
        return "verifier JSON preset does not match this run"
    if verifier_payload.get("mode") != args.mode:
        return "verifier JSON mode does not match this run"
    _, checks_error = _verification_checks_contract(
        verifier_payload.get("checks"),
        run_dir,
        subject="verifier JSON",
    )
    return checks_error


def _run_output_verification(args: argparse.Namespace) -> None:
    if args.preset == "stage0" or args.dry_run or args.skip_output_verification:
        return
    run_dir = _run_dir(args)
    verifier_payload_path = _verification_payload_path(run_dir)
    cmd = _verification_command(args, run_dir)
    res = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    sys.stdout.write(res.stdout)
    sys.stderr.write(res.stderr)
    log_path = run_dir / VERIFICATION_LOG
    json_path = run_dir / VERIFICATION_JSON
    verifier_payload: dict[str, object] | None = None
    verifier_json_error: str | None = None
    if verifier_payload_path.exists():
        try:
            parsed = json.loads(verifier_payload_path.read_text())
        except json.JSONDecodeError as exc:
            verifier_json_error = str(exc)
        else:
            if isinstance(parsed, dict):
                verifier_payload = parsed
            else:
                verifier_json_error = "verifier JSON payload was not an object"
    else:
        verifier_json_error = (
            f"verifier did not write JSON artifact: {verifier_payload_path}"
        )
    verifier_checks = (
        verifier_payload.get("checks")
        if isinstance(verifier_payload, dict)
        else None
    )
    verifier_status = (
        verifier_payload.get("status")
        if isinstance(verifier_payload, dict)
        else None
    )
    verifier_contract_error: str | None = None
    if verifier_json_error is None:
        if verifier_payload is None:
            verifier_contract_error = "verifier JSON payload was not an object"
        else:
            verifier_contract_error = _verifier_payload_contract_error(
                args,
                run_dir,
                verifier_payload,
            )
    _write_text_atomic(
        log_path,
        "\n".join([
            "$ " + shlex.join(cmd),
            "",
            "[stdout]",
            res.stdout.rstrip(),
            "",
            "[stderr]",
            res.stderr.rstrip(),
            "",
        ]),
    )
    verified_artifacts, verified_artifact_error = (
        _completed_verified_artifact_snapshot(run_dir, args.preset)
    )
    record: dict[str, object] = {
        "schema_version": "skinscout.run_verification.v1",
        "status": (
            "ok"
            if (
                res.returncode == 0
                and verifier_json_error is None
                and verifier_contract_error is None
                and verified_artifact_error is None
            )
            else "failed"
        ),
        "verified_at_utc": _utc_now(),
        "command": cmd,
        "returncode": res.returncode,
        "stdout_log": VERIFICATION_LOG,
        "preset": args.preset,
        "mode": args.mode,
        "run_dir": str(run_dir),
        "input_provenance": _input_provenance_from_args(args),
        "diagnostic_nonclaimable_reasons": _diagnostic_nonclaimable_reasons(args),
        "verifier_status": verifier_status,
        "checks": verifier_checks if isinstance(verifier_checks, list) else [],
        "verified_artifacts": verified_artifacts,
    }
    if verifier_payload is not None:
        record["verifier_payload"] = verifier_payload
    if verifier_json_error is not None:
        record["verifier_json_error"] = verifier_json_error
    if verifier_contract_error is not None:
        record["verifier_contract_error"] = verifier_contract_error
    if verified_artifact_error is not None:
        record["verified_artifact_error"] = verified_artifact_error
    _write_json_atomic(json_path, record)
    try:
        verifier_payload_path.unlink()
    except FileNotFoundError:
        pass
    if res.returncode != 0:
        _die("Completed run output verification failed")
    if verifier_json_error is not None:
        _die("Completed run output verification did not produce valid JSON")
    if verifier_contract_error is not None:
        if verifier_status != "ok":
            _die("Completed run output verification JSON status was not ok")
        _die("Completed run output verification JSON contract failed")
    if verified_artifact_error is not None:
        _die("Completed run output verification artifact snapshot failed")


def _run_output_summary(args: argparse.Namespace) -> None:
    if args.preset == "stage0" or args.dry_run or args.skip_output_verification:
        return
    run_dir = _run_dir(args)
    cmd = [
        sys.executable,
        "scripts/summarize_run_outputs.py",
        "--run-dir",
        str(run_dir),
        "--preset",
        args.preset,
        "--mode",
        args.mode,
        "--out-json",
        str(run_dir / "run_summary.json"),
        "--out-md",
        str(run_dir / "run_summary.md"),
        "--target-metadata",
        str(args.target_metadata),
    ]
    if args.allow_safety_degraded:
        cmd.append("--allow-degraded")
    res = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    sys.stdout.write(res.stdout)
    sys.stderr.write(res.stderr)
    if res.returncode != 0:
        _die("Completed run summary generation failed")
    _build_results_viewer(run_dir)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write a small JSON sidecar without leaving a partial file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        temp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temp, path)
    except OSError:
        temp.unlink(missing_ok=True)
        raise


def _build_results_viewer(run_dir: Path) -> None:
    """Write the offline 3D viewer beside the run.

    A wet-lab reader should find it already there rather than having to run a
    command, so this happens on every completed run. Built in-process: it is a
    convenience over results that already exist, and must never fail a finished
    run or disturb the sequence of pipeline subprocesses.
    """
    viewer_dir = run_dir / "viewer"
    status_path = run_dir / "run_viewer_status.json"
    try:
        from make_results_viewer import build as build_viewer

        build_viewer(
            run_dir,
            viewer_dir,
            Path(ROOT / "data" / "hpa" / "proteinatlas.tsv"),
            Path(ROOT / "data" / "human_clean"),
            50,
        )
        print(f"3D 결과 뷰어: {viewer_dir / 'index.html'} (브라우저로 열기)")
        _write_json_atomic(status_path, {"status": "ok", "path": "viewer/index.html"})
    # KeyboardInterrupt must still stop the process; SystemExit is how the
    # builder reports a missing consensus table, so it is an outcome here.
    except (Exception, SystemExit) as exc:  # noqa: BLE001
        reason = str(exc) or exc.__class__.__name__
        sys.stderr.write(
            "3D 결과 뷰어를 만들지 못했습니다(분석 결과 자체는 정상입니다): "
            f"{reason}\n"
        )
        # Recorded rather than left in the log tail: a reader who never sees the
        # viewer otherwise has no way to learn why it is absent.
        _write_json_atomic(status_path, {"status": "failed", "reason": reason})


def _display_value(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _display_list(value: object) -> str:
    if isinstance(value, list):
        items = [str(item) for item in value if str(item).strip()]
        return ", ".join(items) if items else "none"
    return _display_value(value)


def _diagnostic_nonclaimable_reasons(args: argparse.Namespace) -> list[str]:
    reasons: list[str] = []
    skipped = [label for label, _flag in _skipped_readiness_checks(args)]
    if args.allow_unsafe_readiness_skip and skipped:
        reasons.append(
            "readiness preflight skipped with diagnostic override: "
            + ", ".join(skipped)
        )
    if args.skip_output_verification:
        reasons.append("completed-run output verification was skipped")
    if args.allow_safety_degraded:
        reasons.append("explicit degraded ADMET/skin-sens evidence was allowed")
    return reasons


def _readiness_skip_flags_from_reason(reason: str) -> list[str]:
    prefix = "readiness preflight skipped with diagnostic override: "
    if not reason.startswith(prefix):
        return []
    label_to_flag = {
        "data readiness": "--skip-data-readiness",
        "safety readiness": "--skip-safety-readiness",
        "model readiness": "--skip-model-readiness",
    }
    labels = [label.strip() for label in reason.removeprefix(prefix).split(",")]
    return [
        label_to_flag[label]
        for label in labels
        if label in label_to_flag
    ]


def _diagnostic_next_actions(
    args: argparse.Namespace,
    diagnostic_reasons: list[str],
) -> list[str]:
    actions: list[str] = []
    skipped_flags = [flag for _label, flag in _skipped_readiness_checks(args)]
    if not skipped_flags:
        for reason in diagnostic_reasons:
            skipped_flags.extend(_readiness_skip_flags_from_reason(reason))
    if skipped_flags:
        actions.append(
            "rerun without diagnostic readiness skip flags before treating this "
            f"run as claimable: remove {', '.join(dict.fromkeys(skipped_flags))}"
        )
    if args.skip_output_verification or any(
        reason == "completed-run output verification was skipped"
        for reason in diagnostic_reasons
    ):
        actions.append(
            "rerun without --skip-output-verification and keep completed-run "
            "verification enabled before treating output as claimable"
        )
    if args.allow_safety_degraded or any(
        reason == "explicit degraded ADMET/skin-sens evidence was allowed"
        for reason in diagnostic_reasons
    ):
        actions.append(
            "rerun readiness and launcher without --allow-safety-degraded before "
            "treating ADMET/skin-sens output as claimable"
        )
    return list(dict.fromkeys(actions))


def _diagnostic_nonclaimable_reasons_from_record(
    payload: dict[str, object],
) -> list[str]:
    reasons = payload.get("diagnostic_nonclaimable_reasons")
    if not isinstance(reasons, list):
        _die(
            "Completed run verification record missing diagnostic "
            "non-claimable reasons"
        )
    parsed: list[str] = []
    for idx, reason in enumerate(reasons, start=1):
        if not isinstance(reason, str) or not reason.strip():
            _die(
                "Completed run verification record diagnostic "
                f"non-claimable reason {idx} must be a non-empty string"
            )
        parsed.append(reason)
    return parsed


def _combined_diagnostic_nonclaimable_reasons(
    args: argparse.Namespace,
    verification_record: dict[str, object],
) -> list[str]:
    combined: list[str] = []
    seen: set[str] = set()
    for reason in (
        _diagnostic_nonclaimable_reasons_from_record(verification_record)
        + _diagnostic_nonclaimable_reasons(args)
    ):
        if reason in seen:
            continue
        seen.add(reason)
        combined.append(reason)
    return combined


def _validate_completed_verification_command(
    args: argparse.Namespace,
    payload: dict[str, object],
    run_dir: Path,
) -> None:
    command = payload.get("command")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(item, str) and item.strip() for item in command)
    ):
        _die("Completed run verification record command must be a non-empty list")
    expected = _verification_command(args, run_dir)
    observed_tail = command[1:]
    expected_tail = expected[1:]
    if observed_tail not in (expected_tail, [*expected_tail, "--allow-degraded"]):
        _die("Completed run verification record command does not match this run")
    record_reasons = _diagnostic_nonclaimable_reasons_from_record(payload)
    degraded_reason = "explicit degraded ADMET/skin-sens evidence was allowed"
    command_allows_degraded = "--allow-degraded" in command
    record_is_degraded = degraded_reason in record_reasons
    if command_allows_degraded and not record_is_degraded:
        _die(
            "Completed run verification record command allows degraded safety "
            "without diagnostic reason"
        )
    if record_is_degraded and not command_allows_degraded:
        _die(
            "Completed run verification record degraded diagnostic reason "
            "does not match verifier command"
        )


def _validate_completed_verification_log(
    args: argparse.Namespace,
    payload: dict[str, object],
    run_dir: Path,
    checks: list[dict[str, object]],
) -> None:
    command = payload.get("command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        _die("Completed run verification record command must be a non-empty list")
    log_path = run_dir / VERIFICATION_LOG
    try:
        text = log_path.read_text()
    except OSError as exc:
        _die(f"Completed run verification log could not be read: {log_path}: {exc}")
    lines = text.splitlines()
    expected_command = "$ " + shlex.join(command)
    if not lines or lines[0] != expected_command:
        _die("Completed run verification log command does not match record")
    expected_run = f"run_dir={run_dir} preset={args.preset} mode={args.mode}"
    if "SkinScout run output verification: ok" not in text:
        _die("Completed run verification log missing verifier ok summary")
    if expected_run not in text:
        _die("Completed run verification log run metadata does not match this run")
    for check in checks:
        name = str(check["name"])
        path_text = str(check["path"])
        detail = str(check["detail"])
        expected_check = f"[ok] {name}: {path_text}"
        if detail:
            expected_check += f" - {detail}"
        if expected_check not in lines:
            _die(
                "Completed run verification log missing check entry: "
                f"{name}"
            )


def _validate_completed_verification_checks(
    checks: object,
    run_dir: Path,
) -> list[dict[str, object]]:
    validated, error = _verification_checks_contract(
        checks,
        run_dir,
        subject="Completed run verification record",
    )
    if error is not None:
        _die(error)
    return validated


def _validate_completed_verifier_payload(
    args: argparse.Namespace,
    payload: dict[str, object],
    run_dir: Path,
    checks: list[dict[str, object]],
) -> None:
    for key in (
        "verifier_json_error",
        "verifier_contract_error",
        "verified_artifact_error",
    ):
        if key in payload:
            _die(f"Completed run verification record contains {key}")
    verifier_payload = payload.get("verifier_payload")
    if not isinstance(verifier_payload, dict):
        _die("Completed run verification record missing verifier_payload")
    if verifier_payload.get("schema_version") != VERIFIER_OUTPUT_SCHEMA:
        _die("Completed run verification payload has invalid schema_version")
    if verifier_payload.get("status") != "ok":
        _die("Completed run verification payload status is not ok")
    if verifier_payload.get("run_dir") != str(run_dir):
        _die("Completed run verification payload run_dir does not match this run")
    if verifier_payload.get("preset") != args.preset:
        _die("Completed run verification payload preset does not match this run")
    if verifier_payload.get("mode") != args.mode:
        _die("Completed run verification payload mode does not match this run")
    verifier_checks = verifier_payload.get("checks")
    if not isinstance(verifier_checks, list) or not verifier_checks:
        _die("Completed run verification payload checks must be a non-empty list")
    if verifier_checks != checks:
        _die("Completed run verification payload checks do not match record")


def _validate_completed_verification_timestamp(payload: dict[str, object]) -> None:
    value = payload.get("verified_at_utc")
    if not isinstance(value, str) or not value.strip():
        _die("Completed run verification record missing verified_at_utc")
    if not value.endswith("Z"):
        _die(
            "Completed run verification record verified_at_utc must be a UTC "
            "ISO-8601 timestamp"
        )
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        _die(
            "Completed run verification record verified_at_utc must be a UTC "
            "ISO-8601 timestamp"
        )
    if parsed.tzinfo != timezone.utc:
        _die(
            "Completed run verification record verified_at_utc must be a UTC "
            "ISO-8601 timestamp"
        )


def _input_provenance_from_args(args: argparse.Namespace) -> dict[str, str | None]:
    if args.preset == "stage0" and not args.smiles and not args.sdf:
        return {
            "input_type": "smiles",
            "input_smiles": DEFAULT_STAGE0_SMILES,
            "input_canonical_smiles": _canonical_smiles(DEFAULT_STAGE0_SMILES),
            "input_sdf": None,
        }
    if args.smiles and not args.sdf:
        input_smiles = args.smiles.strip()
        return {
            "input_type": "smiles",
            "input_smiles": input_smiles,
            "input_canonical_smiles": _canonical_smiles(input_smiles),
            "input_sdf": None,
        }
    if args.sdf and not args.smiles:
        return {
            "input_type": "sdf",
            "input_smiles": None,
            "input_canonical_smiles": None,
            "input_sdf": str(_validate_sdf(args.sdf)),
        }
    _die("Provide exactly one of --smiles or --sdf")


def _validate_completed_summary_input(
    args: argparse.Namespace,
    payload: dict[str, object],
) -> None:
    expected = _input_provenance_from_args(args)
    compound = payload.get("compound")
    if not isinstance(compound, dict):
        _die("Completed run summary missing compound provenance")
    if compound.get("input_type") != expected["input_type"]:
        _die("Completed run summary input_type does not match this run")
    if expected["input_type"] == "smiles":
        if compound.get("input_smiles") != expected["input_smiles"]:
            _die("Completed run summary input SMILES does not match this run")
        if compound.get("input_canonical_smiles") != expected["input_canonical_smiles"]:
            _die(
                "Completed run summary input canonical SMILES does not match this run"
            )
    else:
        if compound.get("input_smiles") is not None:
            _die("Completed run summary input SMILES must be absent for SDF runs")
        if compound.get("input_canonical_smiles") is not None:
            _die(
                "Completed run summary input canonical SMILES must be absent "
                "for SDF runs"
            )
        if compound.get("input_sdf") != expected["input_sdf"]:
            _die("Completed run summary input SDF does not match this run")


def _validate_completed_verification_input(
    args: argparse.Namespace,
    payload: dict[str, object],
) -> None:
    expected = _input_provenance_from_args(args)
    observed = payload.get("input_provenance")
    if not isinstance(observed, dict):
        _die("Completed run verification record missing input provenance")
    if observed.get("input_type") != expected["input_type"]:
        _die("Completed run verification record input_type does not match this run")
    if expected["input_type"] == "smiles":
        if observed.get("input_smiles") != expected["input_smiles"]:
            _die(
                "Completed run verification record input SMILES does not match "
                "this run"
            )
        if observed.get("input_canonical_smiles") != expected["input_canonical_smiles"]:
            _die(
                "Completed run verification record input canonical SMILES does "
                "not match this run"
            )
    else:
        if observed.get("input_smiles") is not None:
            _die(
                "Completed run verification record input SMILES must be absent "
                "for SDF runs"
            )
        if observed.get("input_canonical_smiles") is not None:
            _die(
                "Completed run verification record input canonical SMILES must "
                "be absent for SDF runs"
            )
        if observed.get("input_sdf") != expected["input_sdf"]:
            _die("Completed run verification record input SDF does not match this run")


def _print_skipped_output_verification_result(args: argparse.Namespace) -> None:
    if args.preset == "stage0" or args.dry_run or not args.skip_output_verification:
        return
    diagnostic_reasons = _diagnostic_nonclaimable_reasons(args)
    input_provenance = _input_provenance_from_args(args)
    print("\nCompleted diagnostic run result:")
    print(f"  run_dir: {_run_dir(args)}")
    print(f"  run_id: {_resolve_run_id(args)}")
    print(f"  preset: {args.preset}")
    print(f"  mode: {args.mode}")
    print(f"  input_type: {_display_value(input_provenance.get('input_type'))}")
    if input_provenance.get("input_smiles") is not None:
        print(f"  input_smiles: {input_provenance['input_smiles']}")
    if input_provenance.get("input_canonical_smiles") is not None:
        print(
            "  input_canonical_smiles: "
            f"{input_provenance['input_canonical_smiles']}"
        )
    if input_provenance.get("input_sdf") is not None:
        print(f"  input_sdf: {input_provenance['input_sdf']}")
    print("  summary_json: skipped")
    print("  summary_md: skipped")
    print("  output_verification: skipped")
    print("  claimable: no")
    print("  diagnostic_nonclaimable: yes")
    if diagnostic_reasons:
        print(f"  diagnostic_reason: {'; '.join(diagnostic_reasons)}")
    for action in _diagnostic_next_actions(args, diagnostic_reasons):
        print(f"  diagnostic_next_action: {action}")
    print("  claim_status: diagnostic run; not claimable")


def _target_entry_label(target: object) -> str | None:
    if not isinstance(target, dict):
        return None
    target_id = target.get("target_id")
    if not isinstance(target_id, str) or not target_id.strip():
        return None
    gene = target.get("gene_symbol")
    protein = target.get("protein_name")
    label = f"{gene} ({target_id})" if isinstance(gene, str) and gene.strip() else target_id
    if isinstance(protein, str) and protein.strip():
        label += f" - {protein}"
    return label


def _top_target_label(payload: dict[str, object]) -> str | None:
    target_prediction = payload.get("target_prediction")
    if not isinstance(target_prediction, dict):
        return None
    top_targets = target_prediction.get("top_targets")
    if not isinstance(top_targets, list) or not top_targets:
        return None
    return _target_entry_label(top_targets[0])


def _print_ranked_target_entry(rank: int, target: object) -> None:
    label = _target_entry_label(target)
    if label is None or not isinstance(target, dict):
        return
    print(f"  ranked_target_{rank}: {label}")
    for field in (
        "final_score",
        "docking_rrf",
        "source_count",
        "sources",
        "skin_score",
        "skin_tier",
        "cell_type_preferred",
        "efficacy",
    ):
        if field not in target:
            continue
        value = target.get(field)
        display = (
            _display_list(value)
            if field in {"sources", "efficacy"}
            else _display_value(value)
        )
        print(f"  ranked_target_{rank}_{field}: {display}")


def _read_completed_verification_record(
    args: argparse.Namespace,
    run_dir: Path,
) -> dict[str, object]:
    path = run_dir / VERIFICATION_JSON
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        _die(f"Completed run verification record could not be read: {path}: {exc}")
    if not isinstance(payload, dict):
        _die(f"Completed run verification record must be a JSON object: {path}")
    if payload.get("schema_version") != "skinscout.run_verification.v1":
        _die("Completed run verification record has invalid schema_version")
    _validate_completed_verification_timestamp(payload)
    if payload.get("status") != "ok":
        _die("Completed run verification record status is not ok")
    returncode = payload.get("returncode")
    if isinstance(returncode, bool) or not isinstance(returncode, int) or returncode != 0:
        _die("Completed run verification record returncode is not 0")
    if payload.get("verifier_status") != "ok":
        _die("Completed run verification record verifier_status is not ok")
    if payload.get("run_dir") != str(run_dir):
        _die("Completed run verification record run_dir does not match this run")
    if payload.get("preset") != args.preset:
        _die("Completed run verification record preset does not match this run")
    if payload.get("mode") != args.mode:
        _die("Completed run verification record mode does not match this run")
    if payload.get("stdout_log") != VERIFICATION_LOG:
        _die("Completed run verification record stdout_log does not match this run")
    _validate_completed_verification_input(args, payload)
    _diagnostic_nonclaimable_reasons_from_record(payload)
    _validate_completed_verification_command(args, payload, run_dir)
    checks = _validate_completed_verification_checks(payload.get("checks"), run_dir)
    _validate_completed_verification_log(args, payload, run_dir, checks)
    _validate_completed_verifier_payload(args, payload, run_dir, checks)
    _validate_completed_verified_artifacts(payload, run_dir, args.preset)
    return payload


def _print_completed_run_result(args: argparse.Namespace) -> None:
    if args.preset == "stage0" or args.dry_run or args.skip_output_verification:
        return
    run_dir = _run_dir(args)
    verification_record = _read_completed_verification_record(args, run_dir)
    summary_json = run_dir / "run_summary.json"
    summary_md = run_dir / "run_summary.md"
    try:
        payload = json.loads(summary_json.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        _die(f"Completed run summary could not be read after verification: {summary_json}: {exc}")
    if not isinstance(payload, dict):
        _die(f"Completed run summary must be a JSON object: {summary_json}")
    _validate_completed_summary_identity(args, run_dir, payload)
    _validate_completed_summary_input(args, payload)

    compound = payload.get("compound")
    compound = compound if isinstance(compound, dict) else {}
    overall = payload.get("overall_decision")
    overall = overall if isinstance(overall, dict) else {}
    target_prediction = payload.get("target_prediction")
    target_prediction = target_prediction if isinstance(target_prediction, dict) else {}
    safety = payload.get("safety")
    safety = safety if isinstance(safety, dict) else {}
    admet_risk = safety.get("admet_risk_assessment")
    admet_risk = admet_risk if isinstance(admet_risk, dict) else {}
    admet_metrics = safety.get("admet_metrics")
    admet_metrics = admet_metrics if isinstance(admet_metrics, dict) else {}
    skin_toxicity = payload.get("skin_toxicity")
    skin_toxicity = skin_toxicity if isinstance(skin_toxicity, dict) else {}
    cosmetic_drug = payload.get("cosmetic_drug")
    cosmetic_drug = cosmetic_drug if isinstance(cosmetic_drug, dict) else {}
    skin_binding = payload.get("skin_specialized_binding")
    skin_binding = skin_binding if isinstance(skin_binding, dict) else {}
    artifacts = payload.get("artifacts")
    artifacts = artifacts if isinstance(artifacts, dict) else {}
    action = overall.get("recommended_action")
    diagnostic_reasons = _combined_diagnostic_nonclaimable_reasons(
        args,
        verification_record,
    )
    overall_pass = overall.get("decision") == "PASS"
    summary_claimable = overall.get("claimable") is True
    no_human_review_required = overall.get("requires_human_review") is False
    skin_context_ok = (
        args.preset not in {"target-id", "report"}
        or skin_binding.get("skin_context_supported") is True
    )
    top_target_skin_context_ok = (
        args.preset not in {"target-id", "report"}
        or skin_binding.get("top_target_skin_context_supported") is True
    )
    claimable = (
        overall_pass
        and action == "proceed"
        and summary_claimable
        and no_human_review_required
        and skin_context_ok
        and top_target_skin_context_ok
        and not diagnostic_reasons
    )

    print("\nCompleted run result:")
    print(f"  summary_json: {summary_json}")
    print(f"  summary_md: {summary_md}")
    print(f"  verification_json: {run_dir / VERIFICATION_JSON}")
    print(f"  verification_log: {run_dir / VERIFICATION_LOG}")
    verification_checks = verification_record.get("checks")
    if isinstance(verification_checks, list):
        print(f"  verification_checks: {len(verification_checks)}")
    verified_artifacts = verification_record.get("verified_artifacts")
    if isinstance(verified_artifacts, list):
        print(f"  verified_artifacts: {len(verified_artifacts)}")
    if args.preset == "report":
        html_report = artifacts.get("html_report")
        if not isinstance(html_report, str) or not html_report.strip():
            _die("Completed report summary missing artifacts.html_report")
        if html_report.strip() != "09_report/index.html":
            _die("Completed report summary artifacts.html_report has unexpected path")
        print(f"  html_report: {run_dir / html_report.strip()}")
    print(f"  run_id: {_display_value(payload.get('run_id'))}")
    print(f"  preset: {_display_value(payload.get('preset'))}")
    print(f"  mode: {_display_value(payload.get('mode'))}")
    print(f"  input_type: {_display_value(compound.get('input_type'))}")
    if compound.get("input_smiles") is not None:
        print(f"  input_smiles: {_display_value(compound.get('input_smiles'))}")
    if compound.get("input_canonical_smiles") is not None:
        print(
            "  input_canonical_smiles: "
            f"{_display_value(compound.get('input_canonical_smiles'))}"
        )
    if compound.get("input_sdf") is not None:
        print(f"  input_sdf: {_display_value(compound.get('input_sdf'))}")
    print(f"  canonical_smiles: {_display_value(compound.get('canonical_smiles'))}")
    print(f"  inchikey: {_display_value(compound.get('inchikey'))}")
    print(f"  overall_decision: {_display_value(overall.get('decision'))}")
    print(f"  recommended_action: {_display_value(action)}")
    print(
        "  requires_human_review: "
        f"{_display_value(overall.get('requires_human_review'))}"
    )
    print(f"  claimable: {_display_value(claimable)}")
    if diagnostic_reasons:
        print("  diagnostic_nonclaimable: yes")
        print(f"  diagnostic_reason: {'; '.join(diagnostic_reasons)}")
        for action in _diagnostic_next_actions(args, diagnostic_reasons):
            print(f"  diagnostic_next_action: {action}")
        print("  claim_status: diagnostic run; not claimable")
    elif claimable:
        print("  claim_status: proceed")
    elif action == "review_before_claim":
        print("  claim_status: human review required before claim")
    elif action == "stop_before_claim":
        print("  claim_status: not claimable; stop before claim")
    elif not overall_pass:
        print("  claim_status: not claimable; overall decision is not PASS")
    elif not no_human_review_required:
        print("  claim_status: human review required before claim")
    elif not summary_claimable:
        print("  claim_status: not claimable; summary claimable is not true")
    elif not skin_context_ok:
        print("  claim_status: not claimable; skin-specialized binding context is not supported")
    elif not top_target_skin_context_ok:
        print("  claim_status: not claimable; top binding target skin context is not supported")
    reasons = overall.get("reasons")
    if isinstance(reasons, list):
        reason_text = "; ".join(
            str(reason) for reason in reasons if str(reason).strip()
        )
        if reason_text:
            print(f"  decision_reasons: {reason_text}")
    print(f"  skin_toxicity: {_display_value(skin_toxicity.get('decision'))}")
    print(f"  skin_toxicity_level: {_display_value(skin_toxicity.get('toxicity_level'))}")
    print(f"  skin_reaction_risk: {_display_value(skin_toxicity.get('skin_reaction_risk_level'))}")
    print(f"  skin_reaction_value: {_display_value(skin_toxicity.get('skin_reaction_value'))}")
    print(
        "  structural_alerts_present: "
        f"{_display_value(skin_toxicity.get('structural_alerts_present'))}"
    )
    print(
        "  structural_alert_flags: "
        f"{_display_list(skin_toxicity.get('structural_alert_flags'))}"
    )
    print(
        "  safety_evidence_degraded: "
        f"{_display_value(skin_toxicity.get('degraded'))}"
    )
    print(
        "  missing_skin_sens_models: "
        f"{_display_list(skin_toxicity.get('missing_models'))}"
    )
    skin_sens_evidence = safety.get("skin_sens_evidence")
    if isinstance(skin_sens_evidence, list):
        for row in skin_sens_evidence:
            if not isinstance(row, dict):
                continue
            model = row.get("model")
            if not isinstance(model, str) or not model.strip():
                continue
            print(f"  skin_sens_{model}_status: {_display_value(row.get('status'))}")
            print(f"  skin_sens_{model}_call: {_display_value(row.get('call'))}")
            print(
                f"  skin_sens_{model}_probability: "
                f"{_display_value(row.get('probability'))}"
            )
    print(f"  high_admet_risk_endpoints: {_display_list(admet_risk.get('high_risk_endpoints'))}")
    print(f"  moderate_admet_risk_endpoints: {_display_list(admet_risk.get('moderate_risk_endpoints'))}")
    for metric in COMPLETED_ADMET_METRICS:
        if metric in admet_metrics:
            print(f"  admet_{metric}: {_display_value(admet_metrics.get(metric))}")
    print(f"  cosmetic_drug_decision: {_display_value(cosmetic_drug.get('decision'))}")
    print(f"  drug_policy: {_display_value(cosmetic_drug.get('drug_policy'))}")
    print(f"  drug_warnings: {_display_value(cosmetic_drug.get('n_warnings'))}")
    top_target = _top_target_label(payload)
    if top_target:
        print(f"  top_target: {top_target}")
    if target_prediction:
        print(f"  target_count: {_display_value(target_prediction.get('n_targets'))}")
        print(
            "  screened_target_count: "
            f"{_display_value(target_prediction.get('screened_target_count'))}"
        )
        screening_counts = target_prediction.get("screening_counts")
        if isinstance(screening_counts, dict):
            for key in sorted(screening_counts):
                print(f"  screening_{key}: {_display_value(screening_counts[key])}")
        print(f"  top_targets_reported: {_display_value(target_prediction.get('top_n'))}")
        top_targets = target_prediction.get("top_targets")
        if isinstance(top_targets, list) and top_targets:
            print("  ranked_targets:")
            for rank, target in enumerate(top_targets, start=1):
                _print_ranked_target_entry(rank, target)
    skin_context = skin_binding.get("skin_context_decision")
    if skin_context is not None:
        print(f"  skin_context: {_display_value(skin_context)}")
        print(
            "  skin_context_supported: "
            f"{_display_value(skin_binding.get('skin_context_supported'))}"
        )
        print(
            "  skin_expression_supported: "
            f"{_display_value(skin_binding.get('skin_expression_supported'))}"
        )
        print(
            "  skin_efficacy_supported: "
            f"{_display_value(skin_binding.get('skin_efficacy_supported'))}"
        )
        print(
            "  top_target_final_score: "
            f"{_display_value(skin_binding.get('top_target_final_score'))}"
        )
        print(
            "  top_target_docking_rrf: "
            f"{_display_value(skin_binding.get('top_target_docking_rrf'))}"
        )
        print(
            "  top_target_source_count: "
            f"{_display_value(skin_binding.get('top_target_source_count'))}"
        )
        print(
            "  top_target_sources: "
            f"{_display_list(skin_binding.get('top_target_sources'))}"
        )
        print(
            "  top_target_skin_score: "
            f"{_display_value(skin_binding.get('top_target_skin_score'))}"
        )
        print(
            "  top_target_skin_tier: "
            f"{_display_value(skin_binding.get('top_target_skin_tier'))}"
        )
        print(
            "  top_target_cell_type_preferred: "
            f"{_display_value(skin_binding.get('top_target_cell_type_preferred'))}"
        )
        print(
            "  top_target_gene_symbol: "
            f"{_display_value(skin_binding.get('top_target_gene_symbol'))}"
        )
        print(
            "  top_target_protein_name: "
            f"{_display_value(skin_binding.get('top_target_protein_name'))}"
        )
        print(
            "  top_target_skin_expression_supported: "
            f"{_display_value(skin_binding.get('top_target_skin_expression_supported'))}"
        )
        print(
            "  top_target_skin_efficacy_supported: "
            f"{_display_value(skin_binding.get('top_target_skin_efficacy_supported'))}"
        )
        print(
            "  top_target_skin_context_supported: "
            f"{_display_value(skin_binding.get('top_target_skin_context_supported'))}"
        )
        print(
            "  top_targets_with_skin_efficacy: "
            f"{_display_value(skin_binding.get('top_targets_with_skin_efficacy'))}"
        )
        print(
            "  top_target_skin_efficacy: "
            f"{_display_list(skin_binding.get('top_target_efficacy'))}"
        )
        most_skin_relevant = _target_entry_label(skin_binding.get("most_skin_relevant_target"))
        if most_skin_relevant:
            print(f"  most_skin_relevant_target: {most_skin_relevant}")
        most_skin_relevant_entry = skin_binding.get("most_skin_relevant_target")
        if isinstance(most_skin_relevant_entry, dict):
            for field in ("gene_symbol", "protein_name"):
                print(
                    f"  most_skin_relevant_{field}: "
                    f"{_display_value(most_skin_relevant_entry.get(field))}"
                )
            for field in (
                "final_score",
                "docking_rrf",
                "source_count",
                "sources",
                "skin_score",
                "skin_tier",
                "cell_type_preferred",
                "efficacy",
            ):
                value = most_skin_relevant_entry.get(field)
                if field in {"sources", "efficacy"}:
                    print(f"  most_skin_relevant_{field}: {_display_list(value)}")
                else:
                    print(f"  most_skin_relevant_{field}: {_display_value(value)}")
    if artifacts:
        source_count = len(artifacts)
        target_source_count = sum(1 for key in artifacts if str(key).startswith("target_"))
        print(f"  source_artifacts: {source_count} listed in run_summary.md")
        print(f"  target_source_artifacts: {target_source_count}")
        for key in sorted(artifacts):
            print(f"  artifact_{key}: {_display_value(artifacts[key])}")


def _verify_existing_run(args: argparse.Namespace) -> int:
    if args.preset == "stage0":
        _die(
            "--verify-existing-run is only for completed safety/target-id/report runs"
        )
    if args.dry_run:
        _die("--verify-existing-run cannot be combined with --dry-run")
    if args.skip_output_verification:
        _die("--verify-existing-run cannot be combined with --skip-output-verification")
    _run_output_summary(args)
    _validate_discovery_daina_metadata(args)
    summary_path = _run_dir(args) / "run_summary.json"
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        _die(f"Completed run summary could not be read for v2 migration: {exc}")
    if not isinstance(summary, dict):
        _die("Completed run summary must be an object for v2 migration")
    manifest_path = _run_dir(args) / RUN_MANIFEST_JSON
    if manifest_path.exists():
        _ensure_run_manifest(args)
    else:
        _ensure_run_manifest(args, legacy_summary=summary)
    _run_output_verification(args)
    _print_completed_run_result(args)
    return 0


def _validate_discovery_daina_metadata(args: argparse.Namespace) -> None:
    profile = _profile_from_args(args)
    if profile.evidence_mode.value != "discovery":
        return
    if args.preset not in {"target-id", "report"} or args.mode not in {"fast", "both"}:
        return
    metadata_path = (
        _run_dir(args)
        / "03_targets"
        / "mode_fast"
        / "daina_zoete_proteome.metadata.json"
    )
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _die(f"Discovery Daina metadata validation failed: {metadata_path}: {exc}")
    if not isinstance(payload, dict):
        _die(f"Discovery Daina metadata must be a JSON object: {metadata_path}")
    if payload.get("schema_version") != "skinscout.daina-run.v1":
        _die(f"Discovery Daina metadata schema mismatch: {metadata_path}")
    if payload.get("evidence_mode") != "leave-query-out":
        _die(
            "Discovery Daina metadata evidence_mode must be leave-query-out: "
            f"{metadata_path}"
        )
    if payload.get("cutoff_date") is not None:
        _die(f"Discovery Daina metadata cutoff_date must be null: {metadata_path}")
    similarity = payload.get("exclude_reference_similarity")
    if (
        isinstance(similarity, bool)
        or not isinstance(similarity, (int, float))
        or not math.isfinite(float(similarity))
        or not 0.0 < float(similarity) <= 1.0
    ):
        _die(
            "Discovery Daina metadata exclude_reference_similarity must be "
            f"finite and in (0, 1]: {metadata_path}"
        )
    if payload.get("score_is_calibrated_probability") is not False:
        _die(
            "Discovery Daina metadata must declare scores as uncalibrated: "
            f"{metadata_path}"
        )

    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        _die(f"Discovery Daina metadata inputs must be an object: {metadata_path}")

    def resolve_input(field: str) -> Path:
        raw = inputs.get(field)
        if not isinstance(raw, str) or not raw.strip():
            _die(f"Discovery Daina metadata inputs.{field} is required: {metadata_path}")
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT / path
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            _die(
                f"Discovery Daina metadata inputs.{field} is unavailable: "
                f"{metadata_path}: {exc}"
            )
        if not resolved.is_file():
            _die(
                f"Discovery Daina metadata inputs.{field} is not a file: "
                f"{metadata_path}"
            )
        return resolved

    def file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    bound_inputs: dict[str, tuple[Path, str]] = {}
    for field in ("ligand_sdf", "fingerprints", "activities"):
        path = resolve_input(field)
        expected_sha256 = inputs.get(f"{field}_sha256")
        if (
            not isinstance(expected_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
            or file_sha256(path) != expected_sha256
        ):
            _die(
                f"Discovery Daina metadata inputs.{field}_sha256 mismatch: "
                f"{metadata_path}"
            )
        bound_inputs[field] = (path, expected_sha256)

    run_dir = _run_dir(args).resolve()
    if not bound_inputs["ligand_sdf"][0].is_relative_to(run_dir):
        _die(
            "Discovery Daina metadata ligand_sdf is outside the active run: "
            f"{metadata_path}"
        )
    fingerprints_path, fingerprints_sha256 = bound_inputs["fingerprints"]
    activities_path, activities_sha256 = bound_inputs["activities"]
    if activities_path != fingerprints_path.parent / "human_activities.parquet":
        _die(
            "Discovery Daina metadata activities/fingerprints snapshot mismatch: "
            f"{metadata_path}"
        )

    snapshot_raw = payload.get("evidence_snapshot_id")
    if not isinstance(snapshot_raw, str) or not snapshot_raw.strip():
        _die(f"Discovery Daina metadata evidence_snapshot_id is required: {metadata_path}")
    snapshot_path = Path(snapshot_raw)
    if not snapshot_path.is_absolute():
        snapshot_path = ROOT / snapshot_path
    try:
        snapshot_path = snapshot_path.resolve(strict=True)
    except OSError as exc:
        _die(f"Discovery Daina evidence snapshot is unavailable: {metadata_path}: {exc}")
    if snapshot_path != fingerprints_path.parent / "source_manifest.json":
        _die(f"Discovery Daina evidence snapshot/path mismatch: {metadata_path}")

    fingerprint_manifest_path = fingerprints_path.parent / "fingerprint_manifest.json"
    try:
        source_manifest = json.loads(snapshot_path.read_text(encoding="utf-8"))
        fingerprint_manifest = json.loads(
            fingerprint_manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        _die(f"Discovery Daina snapshot manifests are unreadable: {metadata_path}: {exc}")
    if not isinstance(source_manifest, dict) or not isinstance(
        fingerprint_manifest, dict
    ):
        _die(f"Discovery Daina snapshot manifests must be objects: {metadata_path}")
    source_outputs = source_manifest.get("output_sha256")
    fingerprint_artifact = fingerprint_manifest.get("artifact")
    fingerprint_input = fingerprint_manifest.get("input")
    fingerprint_source = fingerprint_manifest.get("source_snapshot")
    if (
        source_manifest.get("schema_version") != "chembl_activity_evidence.v1"
        or not isinstance(source_outputs, dict)
        or source_outputs.get("human_activities.parquet") != activities_sha256
        or fingerprint_manifest.get("schema_version")
        != "chembl_fingerprint_snapshot.v1"
        or not isinstance(fingerprint_artifact, dict)
        or fingerprint_artifact.get("sha256") != fingerprints_sha256
        or not isinstance(fingerprint_input, dict)
        or fingerprint_input.get("sha256") != activities_sha256
        or not isinstance(fingerprint_source, dict)
        or fingerprint_source.get("manifest_sha256") != file_sha256(snapshot_path)
    ):
        _die(f"Discovery Daina ChEMBL snapshot binding mismatch: {metadata_path}")

    counts = payload.get("counts")
    if not isinstance(counts, dict):
        _die(f"Discovery Daina metadata counts must be an object: {metadata_path}")
    required_counts = (
        "initial_activity_rows",
        "pre_similarity_filter_activity_rows",
        "scored_activity_rows",
        "excluded_activity_rows_total",
        "ranked_targets",
    )
    if any(
        isinstance(counts.get(field), bool)
        or not isinstance(counts.get(field), int)
        or counts[field] < 0
        for field in required_counts
    ):
        _die(f"Discovery Daina metadata counts are invalid: {metadata_path}")
    if (
        counts["initial_activity_rows"] < counts["pre_similarity_filter_activity_rows"]
        or counts["pre_similarity_filter_activity_rows"] <= 0
        or counts["scored_activity_rows"] <= 0
        or counts["ranked_targets"] <= 0
        or counts["scored_activity_rows"] + counts["excluded_activity_rows_total"]
        != counts["pre_similarity_filter_activity_rows"]
    ):
        _die(f"Discovery Daina metadata count conservation failed: {metadata_path}")


def build_snakemake_command(
    args: argparse.Namespace,
    *,
    validate_conda_frontend: bool,
) -> list[str]:
    _validate_user_config(args.extra_config)
    mode = args.mode
    run_dti_sanity = not args.no_dti_sanity
    targets = _preset_targets(args.preset, mode, run_dti_sanity)

    if args.preset == "stage0" and not args.smiles and not args.sdf:
        input_config = f"compound_smiles={DEFAULT_STAGE0_SMILES}"
    elif bool(args.smiles) == bool(args.sdf):
        _die("Provide exactly one of --smiles or --sdf")
    else:
        if args.smiles:
            input_smiles = args.smiles.strip()
            _canonical_smiles(input_smiles)
            _require_in_scope(assess_applicability(input_smiles))
            input_config = "compound_smiles=" + input_smiles.replace("\\", "\\\\")
        else:
            sdf_path = _validate_sdf(args.sdf)
            _require_in_scope(assess_sdf_applicability(sdf_path))
            input_config = "compound_sdf=" + str(sdf_path).replace("\\", "\\\\")
    run_id = _resolve_run_id(args)
    config = [f"run_id={run_id}", f"mode={mode}", input_config]
    config.append(f"run_dti_sanity={_bool_text(run_dti_sanity)}")
    config.append(f"evidence_mode={_profile_from_args(args).evidence_mode.value}")
    is_discovery = _profile_from_args(args).evidence_mode.value == "discovery"
    immutable_config = IMMUTABLE_DISCOVERY_CONFIG if is_discovery else []
    config.extend(_snakemake_config_entries([
        *args.extra_config,
        *_sota_config_entries(args),
        *immutable_config,
    ]))
    if args.allow_safety_degraded:
        config.extend(_snakemake_config_entries(SAFETY_DEGRADED_CONFIG))
    target_metadata = getattr(args, "target_metadata", DEFAULT_TARGET_METADATA)
    config.append(f"target_metadata={target_metadata}")

    cmd = [
        args.snakemake,
        "-s",
        "workflow/Snakefile",
        "--cores",
        str(args.cores),
    ]
    if args.use_conda:
        cmd.append("--use-conda")
        frontend = _resolve_conda_frontend(
            args.conda_frontend,
            validate=validate_conda_frontend,
        )
        if frontend is not None:
            cmd.extend(["--conda-frontend", frontend])
    if getattr(args, "conda_create_envs_only", False):
        if not args.use_conda:
            _die("--conda-create-envs-only requires --use-conda")
        cmd.append("--conda-create-envs-only")
    if args.dry_run:
        cmd.append("-n")
    cmd.extend(targets)
    for resource in args.resources:
        cmd.extend(["--resources", resource])
    cmd.append("--config")
    cmd.extend(config)
    return cmd


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conda-create-envs-only", action="store_true",
        help="Prepare the exact Snakemake environments without running analysis or writing run results.",
    )
    parser.add_argument("compound", nargs="?", help="Input compound SMILES")
    parser.add_argument("--smiles", help="Input compound SMILES")
    parser.add_argument("--sdf", type=Path, help="Input compound SDF path")
    parser.add_argument(
        "--run-id",
        help="Safe output run identifier; defaults to a deterministic ID from the input",
    )
    parser.add_argument("--mode", choices=sorted(MODES), default="comprehensive")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="target-id",
        help=(
            "Depth to run: safety=ADMET/skin gates, target-id=protein ranking, "
            "report=full workflow; target-id-sota/report-sota enable SOTA "
            "claim thresholds while preserving target-id/report output contracts"
        ),
    )
    parser.add_argument(
        "--sota-claim",
        action="store_true",
        help="Enable SOTA benchmark thresholds/config for target-id/report runs.",
    )
    parser.add_argument(
        "--context-profile",
        choices=sorted(SOTA_CONTEXT_PROFILES),
        default="auto",
        help="Skin context profile recorded for SOTA evidence and evaluation.",
    )
    parser.add_argument(
        "--evidence-mode",
        choices=sorted(EVIDENCE_MODES),
        default="evidence",
        help=(
            "Evidence uses source-backed records; Discovery excludes exact direct "
            "records through the sealed leakage audit."
        ),
    )
    parser.add_argument("--cores", type=int, default=16)
    parser.add_argument(
        "--resources",
        action="append",
        default=["gpu=1"],
        help="Snakemake resource, repeatable; default: gpu=1",
    )
    parser.add_argument("--snakemake", default="snakemake")
    parser.add_argument("--no-use-conda", dest="use_conda", action="store_false")
    parser.set_defaults(use_conda=True)
    parser.add_argument(
        "--conda-frontend",
        choices=["auto", "conda", "mamba"],
        default="auto",
        help="Snakemake conda frontend when --use-conda is enabled",
    )
    parser.add_argument(
        "--no-dti-sanity",
        action="store_true",
        help="Set workflow run_dti_sanity=false for comprehensive/both modes",
    )
    parser.add_argument(
        "--extra-config",
        action="append",
        default=[],
        help="Additional Snakemake config override, key=value; repeatable",
    )
    parser.add_argument(
        "--target-metadata",
        type=Path,
        default=DEFAULT_TARGET_METADATA,
        help="HPA-style TSV mapping UniProt IDs to gene/protein labels.",
    )
    parser.add_argument(
        "--readiness-command",
        help=(
            "Command prefix for model readiness; defaults to the current Demo "
            "runtime, or cosmax-boltz2 when advanced models are required"
        ),
    )
    parser.add_argument(
        "--safety-readiness-command",
        help="Command prefix for safety readiness, default: current Python",
    )
    parser.add_argument(
        "--skip-model-readiness",
        action="store_true",
        help="Skip model/tool readiness preflight before target-id/report runs",
    )
    parser.add_argument(
        "--skip-safety-readiness",
        action="store_true",
        help="Skip ADMET/skin-sens runtime wiring preflight",
    )
    parser.add_argument(
        "--online-safety-readiness",
        action="store_true",
        help="Also probe public safety web-service reachability before running",
    )
    parser.add_argument(
        "--allow-safety-degraded",
        action="store_true",
        help=(
            "Pass explicit degraded ADMET/skin-sens config overrides to Snakemake; "
            "use only for dry/demo runs when model endpoints are unavailable"
        ),
    )
    parser.add_argument(
        "--allow-stage0-build",
        action="store_true",
        help="Allow missing Stage 0 data artifacts to be generated by Snakemake",
    )
    parser.add_argument(
        "--skip-data-readiness",
        action="store_true",
        help="Skip Stage 0 data artifact preflight",
    )
    parser.add_argument(
        "--allow-unsafe-readiness-skip",
        action="store_true",
        help=(
            "Permit readiness skip flags for actual workflow execution; use only "
            "for isolated diagnostics, not claimable prediction runs"
        ),
    )
    parser.add_argument(
        "--skip-output-verification",
        action="store_true",
        help=(
            "Skip automatic completed-run contract verification after Snakemake "
            "success; use only for isolated diagnostics"
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Pass -n to Snakemake")
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="Print the Snakemake command without running preflight or workflow",
    )
    parser.add_argument(
        "--verify-existing-run",
        action="store_true",
        help=(
            "Do not run Snakemake; regenerate summary/verifier artifacts and "
            "print the completed-run result for an existing run directory"
        ),
    )
    parser.add_argument(
        "--discovery-alias-dir",
        default=Path("data/discovery_aliases"),
        type=Path,
        help="Stage 0 exact-alias package used only for Discovery exclusion.",
    )
    args = parser.parse_args(argv)
    if args.compound:
        if args.smiles or args.sdf:
            parser.error("positional SMILES cannot be combined with --smiles or --sdf")
        args.smiles = args.compound
    _normalize_preset_args(args)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.cores < 1:
        _die("--cores must be >= 1")
    if args.sota_claim and args.preset not in {"target-id", "report"}:
        _die("--sota-claim is only valid with target-id/report SOTA runs")
    if args.verify_existing_run and args.print_command:
        _die("--verify-existing-run cannot be combined with --print-command")
    cmd = build_snakemake_command(
        args,
        validate_conda_frontend=not args.print_command,
    )
    if args.print_command:
        print(shlex.join(cmd))
        return 0
    if args.conda_create_envs_only:
        if args.verify_existing_run or args.dry_run:
            _die("--conda-create-envs-only cannot be combined with verification or dry-run")
        env, conda_shim = _snakemake_environment(args)
        try:
            return int(subprocess.run(cmd, cwd=ROOT, env=env, check=False).returncode)
        finally:
            if conda_shim is not None:
                conda_shim.cleanup()
    _validate_evidence_mode_execution(args)
    if args.verify_existing_run:
        return _verify_existing_run(args)
    _validate_readiness_skip_policy(args)
    _validate_output_verification_skip_policy(args)

    if not args.skip_data_readiness:
        _run_data_readiness(
            args.preset,
            args.mode,
            not args.no_dti_sanity,
            allow_stage0_build=args.allow_stage0_build,
        )
        if args.preset in {"target-id", "report"} and not args.allow_stage0_build:
            _run_stage0_claim_quality_readiness()

    if args.preset in SAFETY_PRESETS and not args.skip_safety_readiness:
        _run_safety_readiness(
            use_conda=args.use_conda,
            readiness_command=args.safety_readiness_command,
            online=args.online_safety_readiness,
            allow_degraded=args.allow_safety_degraded,
        )

    if not args.skip_model_readiness:
        requirements = _required_model_readiness(
            args.preset,
            args.mode,
            not args.no_dti_sanity,
            use_conda=args.use_conda,
        )
        _run_model_readiness(requirements, args.readiness_command, use_conda=args.use_conda)

    if _requires_gpu_admission(args.preset, dry_run=args.dry_run):
        _run_gpu_admission()

    if not args.dry_run:
        _ensure_run_manifest(args, cmd)

    env, conda_shim = _snakemake_environment(args)
    try:
        res = subprocess.run(cmd, cwd=ROOT, env=env, check=False)
    finally:
        if conda_shim is not None:
            conda_shim.cleanup()
    if res.returncode != 0:
        return int(res.returncode)
    if not args.dry_run:
        _validate_discovery_daina_metadata(args)
        _finalize_run_manifest(args, cmd)
    if (
        args.preset in {"target-id", "report"}
        and args.allow_stage0_build
        and not args.dry_run
        and not args.skip_data_readiness
    ):
        _run_stage0_claim_quality_readiness()
    _run_output_summary(args)
    _run_output_verification(args)
    _print_skipped_output_verification_result(args)
    _print_completed_run_result(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
