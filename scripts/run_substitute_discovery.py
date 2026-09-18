#!/usr/bin/env python3
"""Run parent target identification, then discover substitute hypotheses."""

from __future__ import annotations

import argparse
import csv
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .compound_applicability import assess, assess_sdf, refusal_message
    from .run_skinscout import (
        SOTA_CONTEXT_PROFILE_OVERRIDES,
        _canonical_smiles,
        _read_completed_verification_record,
        _snakemake_config_entries,
        _validate_completed_summary_identity,
        _validate_completed_summary_input,
        _validate_run_id,
    )
except ImportError:  # Executed as a standalone script.
    from compound_applicability import (  # type: ignore[no-redef]
        assess,
        assess_sdf,
        refusal_message,
    )
    from run_skinscout import (  # type: ignore[no-redef]
        SOTA_CONTEXT_PROFILE_OVERRIDES,
        _canonical_smiles,
        _read_completed_verification_record,
        _snakemake_config_entries,
        _validate_completed_summary_identity,
        _validate_completed_summary_input,
        _validate_run_id,
    )


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "skinscout.substitute_run.v1"
DISCOVERY_SCHEMA = "skinscout.substitute_discovery.v2"
UNIPROT_ACCESSION_RE = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|"
    r"[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?:-[0-9]+)?$"
)
DEFAULT_RESULTS_ROOT = Path(
    os.environ.get("SKINSCOUT_RESULTS_ROOT") or "results/runs"
)
DEFAULT_ALIAS_EVIDENCE = (
    Path("data/discovery_aliases/sources/chembl_aliases.parquet"),
    Path("data/discovery_aliases/sources/gtopdb_aliases.parquet"),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(path: Path, label: str) -> dict[str, Any]:
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"{label} is required and must be non-empty: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if path.exists() or path.is_symlink() or tmp.exists() or tmp.is_symlink():
        raise SystemExit(f"refusing to overwrite pre-existing substitute artifact: {path}")
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"{label} is required and must be non-empty: {path}")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{label} failed to parse: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must be a JSON object: {path}")
    return payload


def _run_dir(results_root: Path, run_id: str) -> Path:
    root = (results_root if results_root.is_absolute() else ROOT / results_root).resolve()
    run_dir = (root / _validate_run_id(run_id)).resolve()
    try:
        run_dir.relative_to(root)
    except ValueError as exc:
        raise SystemExit("run_id escapes the configured results root") from exc
    return run_dir


def _results_root(args: argparse.Namespace) -> Path:
    return (
        args.results_root
        if args.results_root.is_absolute()
        else (ROOT / args.results_root).resolve()
    )


def _normalize_sdf_argument(args: argparse.Namespace) -> None:
    if args.sdf is None:
        return
    try:
        resolved = args.sdf.expanduser().resolve(strict=True)
    except OSError as exc:
        raise SystemExit(f"--sdf must point to a readable non-empty file: {args.sdf}") from exc
    try:
        valid = resolved.is_file() and resolved.stat().st_size > 0
    except OSError as exc:
        raise SystemExit(f"--sdf must point to a readable non-empty file: {args.sdf}") from exc
    if not valid:
        raise SystemExit(f"--sdf must point to a readable non-empty file: {args.sdf}")
    args.sdf = resolved


def build_parent_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_skinscout.py"),
        "--preset",
        "target-id",
        "--mode",
        args.mode,
        "--evidence-mode",
        "evidence",
        "--context-profile",
        args.context_profile,
        "--run-id",
        args.run_id,
        "--cores",
        str(args.cores),
        "--online-safety-readiness",
        "--extra-config",
        f"paths.results_root={_results_root(args)}",
    ]
    if args.smiles:
        command.extend(["--smiles", args.smiles])
    else:
        command.extend(["--sdf", str(args.sdf)])
    if args.no_use_conda:
        command.append("--no-use-conda")
    return command


def _snakemake_executable() -> str:
    available = shutil.which("snakemake")
    if available is not None:
        return available
    sibling = Path(sys.executable).resolve().with_name("snakemake")
    return str(sibling) if sibling.is_file() and os.access(sibling, os.X_OK) else "snakemake"


def build_anchor_command(args: argparse.Namespace) -> list[str]:
    input_config = (
        f"compound_smiles={args.smiles.strip()}"
        if args.smiles
        else f"compound_sdf={args.sdf.resolve()}"
    )
    extra_config = [f"paths.results_root={_results_root(args)}"]
    if args.context_profile != "auto":
        extra_config.append(
            f"evaluation.sota.default_context_profile={args.context_profile}"
        )
        extra_config.extend(SOTA_CONTEXT_PROFILE_OVERRIDES[args.context_profile])
    command = [
        _snakemake_executable(),
        "-s",
        "workflow/Snakefile",
        "--cores",
        str(args.cores),
    ]
    if not args.no_use_conda:
        command.append("--use-conda")
        if shutil.which("conda") is not None:
            command.extend(["--conda-frontend", "conda"])
        elif shutil.which("mamba") is not None or shutil.which("micromamba") is not None:
            command.extend(["--conda-frontend", "mamba"])
    command.extend(
        [
            "pharmacophore_anchor_map",
            "--config",
            f"run_id={args.run_id}",
            f"mode={args.mode}",
            input_config,
            "run_dti_sanity=true",
            "evidence_mode=evidence",
            *_snakemake_config_entries(extra_config),
        ]
    )
    return command


def _ranking_targets(run_dir: Path, summary: dict[str, Any]) -> list[str]:
    artifacts = summary.get("artifacts")
    relative = (
        artifacts.get("target_ranking")
        if isinstance(artifacts, dict)
        else None
    ) or "03_targets/ranked_targets_v3_with_efficacy.csv"
    if not isinstance(relative, str):
        raise SystemExit("Parent summary target ranking path is invalid")
    ranking = (run_dir / relative).resolve()
    try:
        ranking.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise SystemExit("Parent summary target ranking escapes the run directory") from exc
    if not ranking.exists() or not ranking.is_file() or ranking.stat().st_size == 0:
        top_targets = summary.get("target_prediction", {}).get("top_targets", [])
        return [
            str(row.get("target_id", "")).strip()
            for row in top_targets
            if isinstance(row, dict) and str(row.get("target_id", "")).strip()
        ]
    try:
        with ranking.open(encoding="utf-8", newline="", errors="strict") as stream:
            reader = csv.DictReader(stream)
            if "target_id" not in (reader.fieldnames or []):
                raise SystemExit(f"Parent target ranking lacks target_id: {ranking}")
            targets = [str(row.get("target_id") or "").strip() for row in reader]
    except (OSError, UnicodeError, csv.Error) as exc:
        raise SystemExit(f"Parent target ranking failed to parse: {ranking}: {exc}") from exc
    targets = [target for target in targets if target]
    if len(targets) != len(set(targets)):
        raise SystemExit(f"Parent target ranking contains duplicate target_id values: {ranking}")
    return targets


def _validate_parent_run(
    run_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = _read_json(run_dir / "run_summary.json", "Parent run summary")
    raw_verification = _read_json(
        run_dir / "run_verification.json", "Parent run verification"
    )
    command = raw_verification.get("command")
    if not isinstance(command, list) or command.count("--target-metadata") != 1:
        raise SystemExit(
            "Parent run verification command must identify target metadata exactly once"
        )
    metadata_index = command.index("--target-metadata") + 1
    if metadata_index >= len(command) or not isinstance(command[metadata_index], str):
        raise SystemExit("Parent run verification command has invalid target metadata")
    validation_args = argparse.Namespace(**vars(args))
    validation_args.preset = "target-id"
    validation_args.target_metadata = Path(command[metadata_index])
    validation_args.allow_safety_degraded = "--allow-degraded" in command
    verification = _read_completed_verification_record(validation_args, run_dir)
    diagnostic_reasons = verification.get("diagnostic_nonclaimable_reasons")
    if diagnostic_reasons != []:
        raise SystemExit(
            "Parent run verification contains diagnostic non-claimable reasons"
        )
    _validate_completed_summary_identity(validation_args, run_dir, summary)
    _validate_completed_summary_input(validation_args, summary)
    compound = summary.get("compound")
    canonical = compound.get("canonical_smiles") if isinstance(compound, dict) else None
    if not isinstance(canonical, str) or not canonical.strip():
        raise SystemExit("Parent run summary lacks canonical compound SMILES")
    if args.smiles:
        # summary의 canonical_smiles는 Dimorphite가 고른 pH 7.2–7.6 미세상태라
        # 이온화된 입력에서는 사용자가 준 원본과 다르다. 원본 입력(중성화 전)을
        # 기록한 input_canonical_smiles와 비교하고, 없는 구버전 요약에만 기존
        # 비교를 유지한다.
        recorded_raw = compound.get("input_canonical_smiles")
        if isinstance(recorded_raw, str) and recorded_raw.strip():
            if recorded_raw != _canonical_smiles(args.smiles):
                raise SystemExit("Parent run compound does not match the requested SMILES")
        elif canonical != _canonical_smiles(args.smiles):
            raise SystemExit("Parent run compound does not match the requested SMILES")
    else:
        expected_sdf = str(args.sdf)
        if compound.get("input_type") != "sdf" or compound.get("input_sdf") != expected_sdf:
            raise SystemExit("Parent run compound does not match the requested SDF")
    provenance = verification.get("input_provenance")
    if not isinstance(provenance, dict):
        raise SystemExit("Parent run verification lacks input provenance")
    for key in ("input_type", "input_smiles", "input_canonical_smiles", "input_sdf"):
        if provenance.get(key) != compound.get(key):
            raise SystemExit(
                "Parent run verification input provenance does not match the summary"
            )
    targets = _ranking_targets(run_dir, summary)
    if not targets:
        raise SystemExit("Parent run produced no ranked target for substitute discovery")
    return summary, verification


def select_target(
    run_dir: Path,
    summary: dict[str, Any],
    requested_target: str | None,
) -> tuple[str, str]:
    targets = _ranking_targets(run_dir, summary)
    if not targets:
        raise SystemExit("Parent run produced no ranked target for substitute discovery")
    if requested_target:
        target = requested_target.strip().upper()
        if not UNIPROT_ACCESSION_RE.fullmatch(target):
            raise SystemExit("--target-id must be a valid UniProt accession")
        if target not in targets:
            raise SystemExit(
                f"Requested target {target} is absent from the parent target ranking"
            )
        return target, "user_selected_ranked_target"
    target = targets[0]
    if not UNIPROT_ACCESSION_RE.fullmatch(target):
        raise SystemExit(f"Top parent target is not a valid UniProt accession: {target!r}")
    return target, "parent_top_prediction"


def build_discovery_command(
    args: argparse.Namespace,
    *,
    parent_smiles: str,
    target_id: str,
    target_basis: str,
    out_dir: Path,
    interaction_anchor_map: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "discover_substitutes.py"),
        "--parent-smiles",
        parent_smiles,
        "--target-id",
        target_id,
        "--target-selection-basis",
        target_basis,
        "--run-id",
        args.run_id,
        "--candidate-library",
        str(args.candidate_library),
        "--out-dir",
        str(out_dir),
        "--max-candidates",
        str(args.max_candidates),
        "--pharmacophore-track-fraction",
        str(args.pharmacophore_track_fraction),
        "--seed",
        str(args.seed),
    ]
    for path in args.activity_evidence:
        command.extend(["--activity-evidence", str(path)])
    alias_paths = list(args.alias_evidence)
    if not alias_paths:
        alias_paths = [
            path
            for path in DEFAULT_ALIAS_EVIDENCE
            if (ROOT / path).is_file() and (ROOT / path).stat().st_size > 0
        ]
    for path in alias_paths:
        command.extend(["--alias-evidence", str(path)])
    if interaction_anchor_map is not None:
        command.extend(["--interaction-anchor-map", str(interaction_anchor_map)])
    return command


def _run_command(
    command: list[str],
    label: str,
    *,
    env: dict[str, str] | None = None,
) -> None:
    print(f"[{label}] $ {shlex.join(command)}", flush=True)
    result = subprocess.run(command, cwd=ROOT, check=False, env=env)
    if result.returncode != 0:
        raise SystemExit(int(result.returncode))


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _validate_candidate_binding_evidence(
    candidate: dict[str, Any],
    *,
    index: int,
    max_potency_loss: float,
) -> None:
    tier = candidate.get("evidence_tier")
    if tier not in {"direct_retained", "direct_activity", "direct_reduced", "proxy_only"}:
        raise SystemExit(
            f"Substitute discovery candidate {index} evidence tier is invalid"
        )
    activity_count = candidate.get("activity_evidence_count")
    if (
        isinstance(activity_count, bool)
        or not isinstance(activity_count, int)
        or activity_count < 0
        or (tier == "proxy_only" and activity_count != 0)
        or (tier != "proxy_only" and activity_count < 1)
    ):
        raise SystemExit(
            f"Substitute discovery candidate {index} activity evidence count is invalid"
        )
    retained = candidate.get("binding_retained")
    if (
        (tier == "direct_retained" and retained is not True)
        or (tier == "direct_reduced" and retained is not False)
        or (tier in {"direct_activity", "proxy_only"} and retained is not None)
    ):
        raise SystemExit(
            f"Substitute discovery candidate {index} binding-retention flag is invalid"
        )

    comparison_basis = candidate.get("binding_comparison_basis")
    comparison_strata = candidate.get("binding_comparison_strata")
    raw_delta = candidate.get("binding_pactivity_delta")
    binding_delta = _finite_float(raw_delta)
    if tier not in {"direct_retained", "direct_reduced"}:
        if comparison_basis is not None or comparison_strata != [] or raw_delta is not None:
            raise SystemExit(
                f"Substitute discovery candidate {index} binding comparison is invalid"
            )
        return
    if (
        comparison_basis != "same_activity_type_and_source_conservative_delta"
        or not isinstance(comparison_strata, list)
        or not comparison_strata
        or binding_delta is None
    ):
        raise SystemExit(
            f"Substitute discovery candidate {index} binding comparison is invalid"
        )
    stratum_deltas: list[float] = []
    for stratum in comparison_strata:
        if not isinstance(stratum, dict):
            raise SystemExit(
                f"Substitute discovery candidate {index} binding comparison is invalid"
            )
        candidate_value = _finite_float(stratum.get("candidate_median_pactivity"))
        parent_value = _finite_float(stratum.get("parent_median_pactivity"))
        stratum_delta = _finite_float(stratum.get("pactivity_delta"))
        candidate_count = stratum.get("candidate_evidence_count")
        parent_count = stratum.get("parent_evidence_count")
        if (
            not isinstance(stratum.get("activity_type"), str)
            or not stratum["activity_type"]
            or not isinstance(stratum.get("source"), str)
            or not stratum["source"]
            or candidate_value is None
            or parent_value is None
            or stratum_delta is None
            or isinstance(candidate_count, bool)
            or not isinstance(candidate_count, int)
            or candidate_count < 1
            or isinstance(parent_count, bool)
            or not isinstance(parent_count, int)
            or parent_count < 1
            or abs((candidate_value - parent_value) - stratum_delta) > 1e-5
        ):
            raise SystemExit(
                f"Substitute discovery candidate {index} binding comparison is invalid"
            )
        stratum_deltas.append(stratum_delta)
    if (
        abs(binding_delta - min(stratum_deltas)) > 1e-5
        or (tier == "direct_retained") != (binding_delta >= -max_potency_loss)
    ):
        raise SystemExit(
            f"Substitute discovery candidate {index} binding comparison is invalid"
        )


def _validate_discovery_report(
    path: Path,
    *,
    run_id: str,
    target_id: str,
) -> dict[str, Any]:
    report = _read_json(path, "Substitute discovery report")
    if report.get("schema_version") != DISCOVERY_SCHEMA:
        raise SystemExit("Substitute discovery report schema_version is invalid")
    if report.get("run_id") != run_id:
        raise SystemExit("Substitute discovery report run_id does not match")
    target = report.get("target")
    if not isinstance(target, dict) or target.get("target_id") != target_id:
        raise SystemExit("Substitute discovery report target does not match")
    anchor_conditioned = target.get("interaction_anchor_conditioned", False)
    if not isinstance(anchor_conditioned, bool):
        raise SystemExit(
            "Substitute discovery report interaction-anchor status is invalid"
        )
    if (
        report.get("claimable") is not False
        or report.get("hypothesis_only") is not True
        or report.get("wet_lab_required") is not True
    ):
        raise SystemExit("Substitute discovery report must remain non-claimable")
    thresholds = report.get("thresholds")
    max_potency_loss = _finite_float(
        thresholds.get("max_direct_pactivity_loss_for_retention")
        if isinstance(thresholds, dict)
        else None
    )
    if max_potency_loss is None or max_potency_loss < 0.0:
        raise SystemExit("Substitute discovery report binding threshold is invalid")
    candidates = report.get("candidates")
    if not isinstance(candidates, list):
        raise SystemExit("Substitute discovery report candidates must be a list")
    strategy = report.get("selection_strategy")
    if (
        not isinstance(strategy, dict)
        or strategy.get("mode") not in {"balanced_tracks", "global_priority"}
        or not isinstance(strategy.get("selected_counts"), dict)
    ):
        raise SystemExit("Substitute discovery report selection strategy is invalid")
    previous_priority: float | None = None
    previous_global_rank = 0
    for index, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            raise SystemExit(f"Substitute discovery candidate {index} must be an object")
        if (
            candidate.get("claimable") is not False
            or candidate.get("hypothesis_only") is not True
            or candidate.get("wet_lab_required") is not True
        ):
            raise SystemExit(
                f"Substitute discovery candidate {index} violates the claim boundary"
            )
        _validate_candidate_binding_evidence(
            candidate,
            index=index,
            max_potency_loss=max_potency_loss,
        )
        anchor_score = candidate.get("target_conditioned_anchor_score")
        if anchor_conditioned:
            anchor_count = candidate.get("target_conditioned_anchor_count")
            preserved_count = candidate.get(
                "target_conditioned_preserved_anchor_count"
            )
            mapping_count = candidate.get("target_conditioned_mapping_count")
            mapping_ambiguous = candidate.get(
                "target_conditioned_mapping_ambiguous"
            )
            mapping_truncated = candidate.get(
                "target_conditioned_mapping_truncated"
            )
            if (
                _finite_float(anchor_score) is None
                or not 0.0 <= float(anchor_score) <= 1.0
                or isinstance(anchor_count, bool)
                or not isinstance(anchor_count, int)
                or anchor_count < 1
                or isinstance(preserved_count, bool)
                or not isinstance(preserved_count, int)
                or not 0 <= preserved_count <= anchor_count
                or candidate.get("target_conditioned_anchor_basis")
                != "pose_supported_parent_anchor_conservative_mcs_feature_preservation"
                or isinstance(mapping_count, bool)
                or not isinstance(mapping_count, int)
                or mapping_count < 0
                or not isinstance(mapping_ambiguous, bool)
                or mapping_ambiguous != (mapping_count > 1)
                or not isinstance(mapping_truncated, bool)
                or candidate.get("analog_pose_verified") is not False
            ):
                raise SystemExit(
                    f"Substitute discovery candidate {index} interaction-anchor evidence is invalid"
                )
        elif anchor_score is not None:
            raise SystemExit(
                f"Substitute discovery candidate {index} has unexpected interaction-anchor evidence"
            )
        if candidate.get("rank") != index:
            raise SystemExit(f"Substitute discovery candidate {index} rank is invalid")
        admission_bases = candidate.get("admission_bases")
        allowed_admission_bases = {
            "strict_2d_pharmacophore",
            "strict_3d_pharmacophore",
            "strict_target_conditioned_pharmacophore",
            "same_target_activity_feature_family",
        }
        pharmacophore_gate_passed = candidate.get("pharmacophore_gate_passed")
        strict_3d_gate_passed = candidate.get(
            "strict_3d_pharmacophore_gate_passed"
        )
        target_conditioned_strict_gate_passed = candidate.get(
            "target_conditioned_strict_gate_passed"
        )
        if (
            not isinstance(admission_bases, list)
            or not admission_bases
            or len(admission_bases) != len(set(admission_bases))
            or any(value not in allowed_admission_bases for value in admission_bases)
            or not isinstance(pharmacophore_gate_passed, bool)
            or pharmacophore_gate_passed
            != ("strict_2d_pharmacophore" in admission_bases)
            or (
                strict_3d_gate_passed is not None
                and not isinstance(strict_3d_gate_passed, bool)
            )
            or (strict_3d_gate_passed is True)
            != ("strict_3d_pharmacophore" in admission_bases)
            or (
                target_conditioned_strict_gate_passed is not None
                and not isinstance(target_conditioned_strict_gate_passed, bool)
            )
            or (target_conditioned_strict_gate_passed is True)
            != ("strict_target_conditioned_pharmacophore" in admission_bases)
            or (
                target_conditioned_strict_gate_passed is True
                and (
                    not anchor_conditioned
                    or strict_3d_gate_passed is not True
                    or candidate.get("target_conditioned_anchor_gate_passed")
                    is not True
                )
            )
            or (
                "same_target_activity_feature_family" in admission_bases
                and candidate.get("evidence_tier") == "proxy_only"
            )
            or (
                not pharmacophore_gate_passed
                and "same_target_activity_feature_family" not in admission_bases
            )
        ):
            raise SystemExit(
                f"Substitute discovery candidate {index} admission basis is invalid"
            )
        global_rank = candidate.get("global_priority_rank")
        if (
            isinstance(global_rank, bool)
            or not isinstance(global_rank, int)
            or global_rank <= previous_global_rank
        ):
            raise SystemExit(
                f"Substitute discovery candidate {index} global rank is invalid"
            )
        previous_global_rank = global_rank
        priority = candidate.get("priority_score")
        if (
            isinstance(priority, bool)
            or not isinstance(priority, (int, float))
            or not math.isfinite(priority)
            or not 0.0 <= priority <= 1.0
            or (previous_priority is not None and priority > previous_priority)
        ):
            raise SystemExit(
                f"Substitute discovery candidate {index} priority order is invalid"
            )
        previous_priority = float(priority)
        expected_tracks: list[str] = []
        if candidate.get("target_conditioned_strict_gate_passed") is True:
            expected_tracks.append("target_conditioned")
        if candidate.get("evidence_tier") != "proxy_only":
            expected_tracks.append("target_activity")
        if candidate.get("cosing_reference") is True:
            expected_tracks.append("cosmetic_material")
        if not expected_tracks:
            expected_tracks.append("feature_proxy")
        if candidate.get("selection_tracks") != expected_tracks:
            raise SystemExit(
                f"Substitute discovery candidate {index} selection tracks are invalid"
            )
    selected_counts = strategy["selected_counts"]
    expected_selected_counts = {
        "all": len(candidates),
        "target_activity": sum(
            "target_activity" in candidate["selection_tracks"]
            for candidate in candidates
        ),
        "cosmetic_material": sum(
            "cosmetic_material" in candidate["selection_tracks"]
            for candidate in candidates
        ),
        "strict_pharmacophore": sum(
            candidate.get("pharmacophore_gate_passed") is True
            for candidate in candidates
        ),
        "strict_3d_pharmacophore": sum(
            candidate.get("strict_3d_pharmacophore_gate_passed") is True
            for candidate in candidates
        ),
        "strict_target_conditioned": sum(
            candidate.get("target_conditioned_strict_gate_passed") is True
            for candidate in candidates
        ),
        "both": sum(len(candidate["selection_tracks"]) > 1 for candidate in candidates),
    }
    if selected_counts != expected_selected_counts:
        raise SystemExit("Substitute discovery report selected counts are invalid")
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    _normalize_sdf_argument(args)
    # run_skinscout와 같은 입력 범위 계약을 지킨다. --reuse-existing-parent-run으로
    # 부모 실행을 재사용하면 이 스크립트가 유일한 입력 관문이 되므로 여기서 판정한다.
    scope = assess(args.smiles) if args.smiles else assess_sdf(args.sdf)
    message = refusal_message(scope)
    if message:
        raise SystemExit(message)
    if args.evidence_mode != "evidence":
        raise SystemExit(
            "Substitute discovery requires evidence mode because same-target public "
            "activity evidence is part of its ranking contract"
        )
    if args.cores < 1 or args.cores > 256:
        raise SystemExit("--cores must be in [1, 256]")
    if args.max_candidates < 1 or args.max_candidates > 500:
        raise SystemExit("--max-candidates must be in [1, 500]")
    if (
        not math.isfinite(args.pharmacophore_track_fraction)
        or not 0.0 <= args.pharmacophore_track_fraction <= 1.0
    ):
        raise SystemExit("--pharmacophore-track-fraction must be in [0, 1]")
    run_dir = _run_dir(args.results_root, args.run_id)
    out_dir = run_dir / "05_6_substitutes"
    manifest_path = out_dir / "substitute_run_manifest.json"
    expected_outputs = [
        manifest_path,
        *(
            out_dir / filename
            for filename in (
                "substitute_report.json",
                "substitute_candidates.csv",
                "substitute_candidates_3d.sdf",
                "substitute_report.html",
                "substitute_report.md",
            )
        ),
    ]
    for output in expected_outputs:
        if output.exists() or output.is_symlink():
            raise SystemExit(
                f"refusing to overwrite pre-existing substitute artifact: {output}"
            )
    parent_command = build_parent_command(args)
    if not args.reuse_existing_parent_run:
        _run_command(parent_command, "parent-target-run")
    summary, _verification = _validate_parent_run(run_dir, args)
    target_id, target_basis = select_target(run_dir, summary, args.target_id)
    parent_smiles = str(summary["compound"]["canonical_smiles"])
    interaction_anchor_map = args.interaction_anchor_map
    anchor_command: list[str] | None = None
    if args.build_interaction_anchors:
        anchor_command = build_anchor_command(args)
        anchor_env = os.environ.copy()
        anchor_env.setdefault("PYTHONUNBUFFERED", "1")
        with tempfile.TemporaryDirectory(prefix="skinscout-mamba-") as shim_dir:
            if (
                not args.no_use_conda
                and shutil.which("mamba") is None
                and (micromamba := shutil.which("micromamba")) is not None
            ):
                Path(shim_dir, "mamba").symlink_to(Path(micromamba).resolve())
                anchor_env["PATH"] = shim_dir + os.pathsep + anchor_env.get("PATH", "")
            _run_command(anchor_command, "target-conditioned-anchors", env=anchor_env)
        interaction_anchor_map = (
            run_dir / "05_pharmacophore" / "interaction_anchor_map.json"
        )
        if (
            not interaction_anchor_map.is_file()
            or interaction_anchor_map.stat().st_size == 0
        ):
            raise SystemExit(
                "Target-conditioned anchor workflow completed without a usable "
                f"interaction-anchor map: {interaction_anchor_map}"
            )
    elif interaction_anchor_map is None:
        detected = run_dir / "05_pharmacophore" / "interaction_anchor_map.json"
        if detected.is_file() and detected.stat().st_size > 0:
            interaction_anchor_map = detected
    discovery_command = build_discovery_command(
        args,
        parent_smiles=parent_smiles,
        target_id=target_id,
        target_basis=target_basis,
        out_dir=out_dir,
        interaction_anchor_map=interaction_anchor_map,
    )
    _run_command(discovery_command, "substitute-discovery")
    report = _validate_discovery_report(
        out_dir / "substitute_report.json",
        run_id=args.run_id,
        target_id=target_id,
    )
    output_names = (
        "substitute_report.json",
        "substitute_candidates.csv",
        "substitute_candidates_3d.sdf",
        "substitute_report.html",
        "substitute_report.md",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "run_id": args.run_id,
        "claimable": False,
        "hypothesis_only": True,
        "wet_lab_required": True,
        "parent_run": {
            "summary": _fingerprint(run_dir / "run_summary.json", "Parent run summary"),
            "verification": _fingerprint(
                run_dir / "run_verification.json", "Parent run verification"
            ),
            "preset": "target-id",
            "mode": args.mode,
            "canonical_smiles": parent_smiles,
        },
        "target": {
            "target_id": target_id,
            "selection_basis": target_basis,
        },
        "interaction_anchor_map": (
            _fingerprint(interaction_anchor_map, "Interaction-anchor map")
            if interaction_anchor_map is not None
            else None
        ),
        "commands": {
            "parent": parent_command,
            "anchors": anchor_command,
            "discovery": discovery_command,
            "parent_reused": args.reuse_existing_parent_run,
            "anchors_requested": args.build_interaction_anchors,
        },
        "outputs": {
            name: _fingerprint(out_dir / name, f"Substitute artifact {name}")
            for name in output_names
        },
        "candidate_count": len(report["candidates"]),
        "candidate_summary": report.get("summary"),
        "selection_strategy": report.get("selection_strategy"),
    }
    _write_json_atomic(manifest_path, manifest)
    print(
        "[substitute-run] "
        f"target={target_id} candidates={len(report['candidates'])} "
        f"report={out_dir / 'substitute_report.html'} claimable=false"
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--smiles")
    input_group.add_argument("--sdf", type=Path)
    parser.add_argument("--run-id", required=True, type=_validate_run_id)
    parser.add_argument("--mode", choices=("fast", "comprehensive", "both"), default="fast")
    parser.add_argument("--evidence-mode", choices=("evidence", "discovery"), default="evidence")
    parser.add_argument(
        "--context-profile",
        choices=(
            "auto",
            "general_skin",
            "pigmentation",
            "anti_aging",
            "barrier",
            "acne",
            "inflammation",
            "irritation_sensitization",
        ),
        default="auto",
    )
    parser.add_argument("--target-id")
    anchor_group = parser.add_mutually_exclusive_group()
    anchor_group.add_argument(
        "--interaction-anchor-map",
        type=Path,
        help=(
            "Validated Stage 5.5 interaction anchor map; when omitted, the parent "
            "run artifact is used automatically if present"
        ),
    )
    anchor_group.add_argument(
        "--build-interaction-anchors",
        action="store_true",
        help=(
            "Run the Boltz/PLIP/ProLIF Stage 5.5 workflow and require its validated "
            "target-conditioned interaction-anchor map before ranking substitutes"
        ),
    )
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--max-candidates", type=int, default=50)
    parser.add_argument("--pharmacophore-track-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=49242)
    parser.add_argument(
        "--candidate-library",
        type=Path,
        default=Path("data/cosing/cosing.parquet"),
    )
    parser.add_argument("--activity-evidence", action="append", type=Path, default=[])
    parser.add_argument("--alias-evidence", action="append", type=Path, default=[])
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--reuse-existing-parent-run", action="store_true")
    parser.add_argument("--no-use-conda", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
