#!/usr/bin/env python3
"""Aggregate preflight report for a SkinScout SMILES run."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import model_readiness  # noqa: E402
import safety_readiness  # noqa: E402
import stage0_verify  # noqa: E402
from data_readiness import (  # noqa: E402
    data_readiness_payload,
    failed_entries_are_stage0_buildable,
    missing_from_payload,
)
from run_skinscout import (  # noqa: E402
    DEFAULT_STAGE0_SMILES,
    DEFAULT_TARGET_METADATA,
    EVIDENCE_MODES,
    MODES,
    PRESETS,
    SAFETY_PRESETS,
    SOTA_CONTEXT_PROFILES,
    _canonical_smiles,
    _normalize_preset_args,
    _required_model_readiness,
    _run_manifest_payload,
    _run_dir,
    _resolve_run_id,
    _validate_run_id,
    build_snakemake_command,
    validate_discovery_alias_package,
)


FAST_TARGET_ARTIFACTS = {
    "target_fast_daina_zoete": "03_targets/mode_fast/daina_zoete_proteome.tsv",
    "target_fast_daina_selected": "03_targets/mode_fast/daina_top256.csv",
    "target_fast_dti_rrf": "03_targets/mode_fast/dti_rrf_top25pct.csv",
    "target_fast_autogrid_manifest": "03_targets/mode_fast/autogrid_map_manifest.json",
    "target_fast_autodock": "03_targets/mode_fast/autodock_top5k.tsv",
    "target_fast_gnina_pose": "03_targets/mode_fast/gnina_pose_rescores.tsv",
    "target_fast_daina_structural_targets": "03_targets/mode_fast/daina_structural_targets.csv",
    "target_fast_rerank_consensus": "03_targets/mode_fast/top50.csv",
}
COMPREHENSIVE_TARGET_ARTIFACTS = {
    "target_comprehensive_ligand": "03_targets/mode_comprehensive/ligand.pdbqt",
    "target_comprehensive_autodock": "03_targets/mode_comprehensive/autodock_all_targets.tsv",
    "target_comprehensive_pre_rescore": "03_targets/mode_comprehensive/top_pct_pre_rescore.csv",
    "target_comprehensive_gnina": "03_targets/mode_comprehensive/gnina_rescores.tsv",
    "target_comprehensive_rtmscore": "03_targets/mode_comprehensive/rtmscore_rescores.tsv",
    "target_comprehensive_boltz2": "03_targets/mode_comprehensive/boltz2_affinity_top.tsv",
    "target_comprehensive_consensus": "03_targets/mode_comprehensive/top50_4way_consensus.csv",
}


def _blocker(label: str, path: Path) -> dict[str, str]:
    return {"label": label, "path": str(path)}


def _status_from_blockers(blockers: list[dict[str, Any]]) -> str:
    return "ok" if not blockers else "failed"


def _data_group(
    *,
    name: str,
    preset: str,
    mode: str,
    run_dti_sanity: bool,
    allow_stage0_build: bool,
    repo_root: Path,
) -> dict[str, Any]:
    payload = data_readiness_payload(
        preset,
        mode,
        run_dti_sanity,
        root=repo_root,
    )
    missing = missing_from_payload(payload)
    blockers = [_blocker(label, path) for label, path in missing]
    status = _status_from_blockers(blockers)
    buildable = (
        bool(blockers)
        and allow_stage0_build
        and failed_entries_are_stage0_buildable(missing)
    )
    if buildable:
        status = "buildable"
    return {
        "name": name,
        "kind": "data",
        "status": status,
        "preset": preset,
        "mode": mode,
        "blocking": status == "failed",
        "blockers": blockers,
        "raw_status": payload["status"],
        "checks": payload["checks"],
    }


def _safety_group(
    *,
    runtime_mode: str,
    online: bool,
    allow_degraded: bool,
) -> dict[str, Any]:
    if runtime_mode == "conda":
        checks = safety_readiness._check_env_manifest(  # noqa: SLF001
            ROOT / "envs/dti.yml",
            allow_degraded=allow_degraded,
        )
    else:
        checks = safety_readiness._check_active_modules(  # noqa: SLF001
            allow_degraded=allow_degraded,
        )
    checks.extend(safety_readiness._check_stage2_endpoints())  # noqa: SLF001
    if online:
        online_checks = safety_readiness._check_online()  # noqa: SLF001
        if allow_degraded:
            for check in online_checks:
                if not check["ok"]:
                    check["ok"] = True
                    check["warning"] = True
        checks.extend(online_checks)
    blockers = [
        {
            "label": str(check["name"]),
            "detail": str(check.get("reason", "")),
        }
        for check in checks
        if not check["ok"]
    ]
    return {
        "name": "safety",
        "kind": "safety",
        "status": _status_from_blockers(blockers),
        "blocking": bool(blockers),
        "runtime_mode": runtime_mode,
        "blockers": blockers,
        "checks": checks,
    }


def _model_group(
    *,
    preset: str,
    mode: str,
    run_dti_sanity: bool,
    use_conda: bool,
    readiness_command: str | None,
) -> dict[str, Any]:
    required = _required_model_readiness(
        preset,
        mode,
        run_dti_sanity,
        use_conda=use_conda,
    )
    cmd = _model_readiness_base_command(readiness_command)
    cmd.append("scripts/model_readiness.py")
    res = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    try:
        payload = json.loads(res.stdout)
    except json.JSONDecodeError:
        payload = {
            "status": "failed",
            "stdout_tail": res.stdout[-1200:],
            "stderr_tail": res.stderr[-1200:],
        }
    if res.returncode != 0:
        payload.setdefault("stderr_tail", res.stderr[-1200:])
    failures = model_readiness.failed_requirements(payload, required)
    if res.returncode != 0 and not failures:
        failures.append(
            {
                "requirement": "model_readiness_command",
                "actual": {
                    "returncode": res.returncode,
                    "stderr_tail": res.stderr[-1200:],
                },
            }
        )
    blockers = [
        {
            "label": str(failure["requirement"]),
            "detail": json.dumps(failure["actual"], sort_keys=True),
        }
        for failure in failures
    ]
    return {
        "name": "model",
        "kind": "model",
        "status": _status_from_blockers(blockers),
        "blocking": bool(blockers),
        "required": required,
        "command": cmd,
        "blockers": blockers,
        "readiness": payload,
    }


def _model_readiness_base_command(readiness_command: str | None) -> list[str]:
    if readiness_command:
        return shlex.split(readiness_command)
    if shutil.which("micromamba"):
        return ["micromamba", "run", "-n", "cosmax-boltz2", "python"]
    return [sys.executable]


def _stage0_claim_quality_group(*, repo_root: Path) -> dict[str, Any]:
    checks = stage0_verify.collect_checks(
        repo_root.resolve(),
        claim_quality=True,
        check_stage0_flag=True,
    )
    blockers = [
        {
            "label": check.name,
            "detail": check.detail,
        }
        for check in checks
        if not check.ok
    ]
    return {
        "name": "stage0_claim_quality",
        "kind": "stage0_claim_quality",
        "status": _status_from_blockers(blockers),
        "blocking": bool(blockers),
        "command": [
            sys.executable,
            "scripts/stage0_verify.py",
            "--strict",
            "--claim-quality",
        ],
        "blockers": blockers,
        "checks": [
            {"name": check.name, "ok": check.ok, "detail": check.detail}
            for check in checks
        ],
    }


def _discovery_alias_group(alias_dir: Path, repo_root: Path = ROOT) -> dict[str, Any]:
    reference = alias_dir / "direct_exact_reference.smi"
    manifest = alias_dir / "manifest.json"
    blockers: list[dict[str, str]] = []
    checks: list[dict[str, Any]] = []
    try:
        canonical_alias_dir = (repo_root / "data" / "discovery_aliases").resolve()
        if alias_dir.resolve() == canonical_alias_dir:
            check = stage0_verify.chk_discovery_alias_integrity(repo_root)
            if not check.ok:
                raise ValueError(check.detail)
        else:
            validate_discovery_alias_package(reference, manifest)
    except (OSError, ValueError) as exc:
        blockers.append({
            "label": "discovery_alias_manifest",
            "path": str(manifest),
            "detail": str(exc),
        })
    checks.append({
        "name": "canonical_stage0_discovery_alias_package",
        "ok": not blockers,
        "reference": str(reference),
        "manifest": str(manifest),
    })
    return {
        "name": "discovery-aliases",
        "kind": "contract",
        "status": _status_from_blockers(blockers),
        "blocking": bool(blockers),
        "blockers": blockers,
        "checks": checks,
    }


def _runtime_claims_gpu(args: argparse.Namespace) -> bool:
    return args.runtime_mode == "gpu" or os.environ.get("SKINSCOUT_GPU_MODE") == "1"


def _gpu_runtime_group() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    blockers: list[dict[str, str]] = []

    nvidia = subprocess.run(
        ["nvidia-smi"],
        text=True,
        capture_output=True,
        check=False,
    )
    nvml_ok = nvidia.returncode == 0 and "Driver/library version mismatch" not in (
        nvidia.stdout + nvidia.stderr
    )
    checks.append({
        "name": "host_nvidia_smi_nvml",
        "ok": nvml_ok,
        "returncode": nvidia.returncode,
        "stderr_tail": nvidia.stderr[-600:],
    })
    if not nvml_ok:
        blockers.append({
            "label": "nvml_driver_library",
            "detail": "nvidia-smi failed or reported Driver/library version mismatch",
        })

    cuda_code = (
        "import torch; "
        "assert torch.cuda.is_available(), 'torch cuda unavailable'; "
        "x=torch.ones((256,256),device='cuda'); "
        "y=(x @ x).sum().item(); "
        "assert y > 0"
    )
    cuda = subprocess.run(
        [sys.executable, "-c", cuda_code],
        text=True,
        capture_output=True,
        check=False,
    )
    cuda_ok = cuda.returncode == 0
    checks.append({
        "name": "real_cuda_tensor_smoke",
        "ok": cuda_ok,
        "returncode": cuda.returncode,
        "stderr_tail": cuda.stderr[-600:],
    })
    if not cuda_ok:
        blockers.append({
            "label": "real_cuda_tensor_smoke",
            "detail": "GPU mode requires an actual CUDA tensor/kernel smoke test",
        })

    return {
        "name": "runtime:gpu",
        "kind": "runtime",
        "status": _status_from_blockers(blockers),
        "blocking": bool(blockers),
        "blockers": blockers,
        "checks": checks,
    }


def _input_group(args: argparse.Namespace) -> tuple[dict[str, Any], str | None]:
    blockers: list[dict[str, Any]] = []
    canonical: str | None = None
    run_id: str | None = None
    input_type: str | None = None
    input_smiles: str | None = None
    input_canonical_smiles: str | None = None
    input_sdf: str | None = None
    if args.run_id:
        try:
            run_id = _validate_run_id(args.run_id)
        except SystemExit as exc:
            blockers.append({"label": "run_id", "detail": str(exc)})

    if args.preset == "stage0" and not args.smiles and not args.sdf:
        input_type = "smiles"
        input_smiles = DEFAULT_STAGE0_SMILES
        canonical = "C"
        input_canonical_smiles = canonical
    elif bool(args.smiles) == bool(args.sdf):
        blockers.append(
            {
                "label": "compound_input",
                "detail": "provide exactly one of --smiles or --sdf",
            }
        )
    elif args.smiles:
        input_type = "smiles"
        input_smiles = args.smiles.strip()
        try:
            canonical = _canonical_smiles(input_smiles)
            input_canonical_smiles = canonical
        except SystemExit as exc:
            blockers.append({"label": "smiles", "detail": str(exc)})
    elif args.sdf is not None:
        input_type = "sdf"
        input_sdf = str(args.sdf)
        if not args.sdf.exists() or args.sdf.stat().st_size == 0:
            blockers.append(
                {
                    "label": "sdf",
                    "detail": f"--sdf must point to a non-empty SDF file: {args.sdf}",
                }
            )
    if not blockers:
        try:
            run_id = _resolve_run_id(args)
        except SystemExit as exc:
            blockers.append({"label": "run_id", "detail": str(exc)})

    return (
        {
            "name": "input",
            "kind": "input",
            "status": _status_from_blockers(blockers),
            "blocking": bool(blockers),
            "run_id": run_id if run_id is not None else args.run_id,
            "input_type": input_type,
            "input_smiles": input_smiles,
            "input_canonical_smiles": input_canonical_smiles,
            "input_sdf": input_sdf,
            "smiles": args.smiles,
            "canonical_smiles": canonical,
            "sdf": str(args.sdf) if args.sdf is not None else None,
            "blockers": blockers,
        },
        canonical,
    )


def _runner_namespace(args: argparse.Namespace) -> Namespace:
    return Namespace(
        smiles=args.smiles,
        sdf=args.sdf,
        run_id=args.run_id,
        mode=args.mode,
        preset=args.preset,
        cores=args.cores,
        resources=args.resources,
        snakemake=args.snakemake,
        use_conda=args.use_conda,
        conda_frontend=args.conda_frontend,
        no_dti_sanity=args.no_dti_sanity,
        extra_config=args.extra_config,
        target_metadata=args.target_metadata,
        allow_safety_degraded=args.allow_safety_degraded,
        sota_claim=args.sota_claim,
        context_profile=args.context_profile,
        evidence_mode=args.evidence_mode,
        discovery_alias_dir=args.discovery_alias_dir,
        requested_preset=getattr(args, "requested_preset", args.preset),
        run_profile=args.run_profile,
        dry_run=args.dry_run,
    )


def _snakemake_command(args: argparse.Namespace) -> str | None:
    try:
        cmd = build_snakemake_command(
            _runner_namespace(args),
            validate_conda_frontend=False,
        )
    except SystemExit:
        return None
    return shlex.join(cmd)


def _compound_field_contract(input_type: str | None) -> list[str]:
    fields = ["input_type"]
    if input_type == "sdf":
        fields.append("input_sdf")
    else:
        fields.extend(["input_smiles", "input_canonical_smiles"])
    fields.extend(["canonical_smiles", "inchikey"])
    return fields


def _summary_field_contract(
    preset: str,
    mode: str,
    input_type: str | None,
) -> dict[str, list[str]]:
    fields = {
        "compound": _compound_field_contract(input_type),
        "safety": [
            "skin_sens_decision",
            "skin_sens_calls",
            "skin_sens_calls.husspred",
            "skin_sens_calls.stoptox",
            "skin_sens_calls.pred_skin",
            "skin_sens_evidence[].model",
            "skin_sens_evidence[].status",
            "skin_sens_evidence[].call",
            "skin_sens_evidence[].probability",
            "admet_metrics",
            "admet_metrics.AMES",
            "admet_metrics.ClinTox",
            "admet_metrics.DILI",
            "admet_metrics.Skin_Reaction",
            "admet_metrics.hERG",
            "admet_metrics.LD50_Zhu",
            "admet_metrics.Solubility_AqSolDB",
            "admet_metrics.logP",
            "admet_metrics.QED",
            "admet_metrics.tpsa",
            "admet_risk_assessment",
            "structural_alert_flags",
            "degraded",
            "missing_models",
        ],
        "skin_toxicity": [
            "decision",
            "toxicity_level",
            "skin_sens_decision",
            "skin_reaction_risk_level",
            "skin_reaction_value",
            "structural_alerts_present",
            "structural_alert_flags",
            "degraded",
            "missing_models",
            "reasons",
        ],
        "cosmetic_drug": [
            "decision",
            "drug_policy",
            "cosing_level",
            "inci",
            "cosing_functions",
            "cosing_tanimoto",
            "max_tanimoto_to_approved_drug",
            "n_warnings",
        ],
        "overall_decision": [
            "decision",
            "requires_human_review",
            "recommended_action",
            "claimable",
            "reasons",
        ],
        "artifacts": [
            "compound",
            "admet_report",
            "admet_ai",
            "structural_alerts",
            "husspred",
            "stoptox",
            "pred_skin",
            "skin_sens_decision",
            "cosing_match",
            "drug_warnings",
            "cosmetic_drug_decision",
        ],
    }
    if preset in {"target-id", "report"}:
        fields["target_prediction"] = [
            "ranking_path",
            "n_targets",
            "screened_target_count",
            "screening_counts",
            "top_n",
            "top_targets[].target_id",
            "top_targets[].gene_symbol",
            "top_targets[].protein_name",
            "top_targets[].final_score",
            "top_targets[].skin_score",
            "top_targets[].skin_tier",
            "top_targets[].docking_rrf",
            "top_targets[].source_count",
            "top_targets[].sources",
            "top_targets[].efficacy",
        ]
        fields["skin_specialized_binding"] = [
            "context",
            "ranking_path",
            "skin_context_decision",
            "skin_context_supported",
            "skin_expression_supported",
            "skin_efficacy_supported",
            "skin_context_reasons",
            "top_target_id",
            "top_target_gene_symbol",
            "top_target_protein_name",
            "top_target_final_score",
            "top_target_docking_rrf",
            "top_target_source_count",
            "top_target_sources",
            "top_target_skin_score",
            "top_target_skin_tier",
            "top_target_efficacy",
            "top_target_skin_expression_supported",
            "top_target_skin_efficacy_supported",
            "top_target_skin_context_supported",
            "top_targets_with_skin_efficacy",
            "most_skin_relevant_target",
            "most_skin_relevant_target.gene_symbol",
            "most_skin_relevant_target.protein_name",
            "most_skin_relevant_target.final_score",
            "most_skin_relevant_target.docking_rrf",
            "most_skin_relevant_target.source_count",
            "most_skin_relevant_target.sources",
            "most_skin_relevant_target.skin_score",
            "most_skin_relevant_target.skin_tier",
            "most_skin_relevant_target.efficacy",
        ]
        fields["overall_decision"].extend(["top_target_id", "target_count"])
        fields["artifacts"].append("target_ranking")
        if mode in {"fast", "both"}:
            fields["artifacts"].extend(FAST_TARGET_ARTIFACTS)
        if mode in {"comprehensive", "both"}:
            fields["artifacts"].extend(COMPREHENSIVE_TARGET_ARTIFACTS)
    if preset == "report":
        fields["artifacts"].append("html_report")
    return fields


def _input_report_markers(input_type: str | None) -> list[str]:
    markers = ["Input type"]
    if input_type == "sdf":
        markers.append("Input SDF")
    else:
        markers.extend(["Input SMILES", "Input canonical SMILES"])
    return markers


def _report_marker_contract(preset: str, input_type: str | None) -> list[str]:
    if preset != "report":
        return []
    return [
        *_input_report_markers(input_type),
        "InChIKey",
        "canonical SMILES",
        "Overall decision",
        "Recommended action",
        "Claimable",
        "Claim status",
        "Skin toxicity",
        "Skin_Reaction",
        "Skin-sens 3-model evidence",
        "Structural alerts",
        "degraded skin-sens evidence",
        "missing skin-sens models",
        "Cosmetic/drug decision",
        "High ADMET risk endpoints",
        "Moderate ADMET risk endpoints",
        "Summary ADMET metrics",
        "Drug-avoidance warnings",
        "Skin context decision",
        "Skin context supported",
        "Skin expression supported",
        "Skin efficacy supported",
        "Screened target candidates",
        "Screening stage counts",
        "Top binding target",
        "top target gene/protein labels when available",
        "top target final score",
        "top target docking RRF when available",
        "top target source count",
        "top target sources",
        "top target skin score",
        "top target skin tier",
        "top target skin efficacy",
        "top target skin context supported",
        "Most skin-relevant top target",
        "most skin-relevant gene/protein labels when available",
        "most skin-relevant score/source/skin-efficacy evidence",
        "top target row details from target_prediction.top_targets",
    ]


def _launcher_result_field_contract(preset: str, input_type: str | None) -> list[str]:
    if preset == "stage0":
        return []
    fields = [
        "summary_json",
        "summary_md",
        "verification_json",
        "verification_log",
        "verification_checks",
        "verified_artifacts",
        "run_id",
        "preset",
        "mode",
        "input_type",
    ]
    if input_type == "sdf":
        fields.append("input_sdf")
    else:
        fields.extend(["input_smiles", "input_canonical_smiles"])
    fields.extend([
        "canonical_smiles",
        "inchikey",
        "overall_decision",
        "recommended_action",
        "requires_human_review",
        "claimable",
        "claim_status",
        "decision_reasons",
        "skin_toxicity",
        "skin_toxicity_level",
        "skin_reaction_risk",
        "skin_reaction_value",
        "structural_alerts_present",
        "structural_alert_flags",
        "safety_evidence_degraded",
        "missing_skin_sens_models",
        "skin_sens_<model>_status",
        "skin_sens_<model>_call",
        "skin_sens_<model>_probability",
        "high_admet_risk_endpoints",
        "moderate_admet_risk_endpoints",
        "admet_<metric>",
        "cosmetic_drug_decision",
        "drug_policy",
        "drug_warnings",
        "source_artifacts",
        "target_source_artifacts",
        "artifact_<key>",
    ])
    if preset in {"target-id", "report"}:
        fields.extend([
            "target_count",
            "screened_target_count",
            "screening_<stage>",
            "top_target",
            "top_targets_reported",
            "ranked_targets",
            "ranked_target_<rank>",
            "ranked_target_<rank>_final_score",
            "ranked_target_<rank>_docking_rrf",
            "ranked_target_<rank>_source_count",
            "ranked_target_<rank>_sources",
            "ranked_target_<rank>_skin_score",
            "ranked_target_<rank>_skin_tier",
            "ranked_target_<rank>_efficacy",
            "skin_context",
            "skin_context_supported",
            "skin_expression_supported",
            "skin_efficacy_supported",
            "top_target_final_score",
            "top_target_docking_rrf",
            "top_target_source_count",
            "top_target_sources",
            "top_target_skin_score",
            "top_target_skin_tier",
            "top_target_gene_symbol",
            "top_target_protein_name",
            "top_target_skin_expression_supported",
            "top_target_skin_efficacy_supported",
            "top_target_skin_context_supported",
            "top_targets_with_skin_efficacy",
            "top_target_skin_efficacy",
            "most_skin_relevant_target",
            "most_skin_relevant_gene_symbol",
            "most_skin_relevant_protein_name",
            "most_skin_relevant_final_score",
            "most_skin_relevant_docking_rrf",
            "most_skin_relevant_source_count",
            "most_skin_relevant_sources",
            "most_skin_relevant_skin_score",
            "most_skin_relevant_skin_tier",
            "most_skin_relevant_efficacy",
        ])
    if preset == "report":
        fields.append("html_report")
    return fields


def _source_artifact_contract(preset: str, mode: str) -> list[str]:
    artifacts = [
        "01_input/compound_canonical.json",
        "02_admet/admet_report.json",
        "02_admet/admet_ai.json",
        "02_admet/structural_alerts.json",
        "02_admet/husspred.json",
        "02_admet/stoptox.json",
        "02_admet/pred_skin.json",
        "02_admet/skin_sens_decision.txt",
        "02b_cosmetic_drug/cosing_match.json",
        "02b_cosmetic_drug/drug_warnings.json",
        "02b_cosmetic_drug/cosmetic_drug_decision.txt",
    ]
    if preset in {"target-id", "report"}:
        artifacts.append("03_targets/ranked_targets_v3_with_efficacy.csv")
        if mode in {"fast", "both"}:
            artifacts.extend(FAST_TARGET_ARTIFACTS.values())
        if mode in {"comprehensive", "both"}:
            artifacts.extend(COMPREHENSIVE_TARGET_ARTIFACTS.values())
    if preset == "report":
        artifacts.append("09_report/index.html")
    return artifacts


def _source_evidence_invariant_contract() -> list[str]:
    return [
        (
            "01_input/compound_canonical.json canonical_smiles must equal "
            "RDKit canonical SMILES"
        ),
        (
            "01_input/compound_canonical.json inchikey must equal "
            "RDKit InChIKey for canonical_smiles"
        ),
        (
            "ADMET-AI, structural-alert, HuSSPred, STopTox, and Pred-Skin "
            "source smiles must canonical-match "
            "01_input/compound_canonical.json canonical_smiles"
        ),
        (
            "ADMET-AI predictions, structural-alert flags/matches, and "
            "skin-sens model calls must agree with 02_admet/admet_report.json"
        ),
    ]


def _target_evidence_invariant_contract(preset: str, mode: str) -> list[str]:
    if preset not in {"target-id", "report"}:
        return []

    min_source_count = 1 if mode == "fast" else 3
    invariants = [
        (
            "03_targets/ranked_targets_v3_with_efficacy.csv must contain "
            "nonblank unique target_id values"
        ),
        (
            "target ranking final_score, skin_score, docking_rrf, and "
            "efficacy_score values must be numeric fractions when present"
        ),
        "target ranking final_score must be sorted descending",
        (
            "target ranking source_count must be an integer >= "
            f"{min_source_count} and match the sources label list"
        ),
        (
            "top ranked targets must include efficacy_top* KG evidence annotation columns; "
            + (
                "blank labels remain explicit unavailable evidence in fast mode"
                if mode == "fast"
                else "top rows must carry efficacy labels"
            )
        ),
        (
            "run_summary target_prediction and skin_specialized_binding fields "
            "must match the target ranking rows"
        ),
        (
            "run_summary target_prediction.screening_counts must match "
            "mode-specific intermediate target evidence row counts"
        ),
        (
            "claimable target-id/report outputs require "
            "skin_specialized_binding.skin_context_supported true and "
            "skin_specialized_binding.top_target_skin_context_supported true"
        ),
    ]
    if mode in {"fast", "both"}:
        invariants.extend([
            (
                "fast target intermediates must include Daina-Zoete proteome "
                "scores, the fixed Daina primary set, explicit AutoGrid maps, "
                "AutoDock poses, actual-pose GNINA annotations, the canonical "
                "overlay, and its top50 compatibility projection"
            ),
            (
                "fast Daina-primary rows must preserve rank/score and have "
                "source_count >= 1 with matching annotation-source labels"
            ),
        ])
    if mode in {"comprehensive", "both"}:
        invariants.extend([
            (
                "comprehensive intermediates must include ligand PDBQT, "
                "AutoDock/Vina, GNINA, RTMScore, Boltz-2, and four-way "
                "consensus evidence"
            ),
            (
                "comprehensive four-way consensus rows must have "
                "source_count >= 3 with matching source labels"
            ),
        ])
    return invariants


def _skipped_readiness_checks(args: argparse.Namespace) -> list[tuple[str, str]]:
    return [
        (label, flag)
        for label, flag, enabled in (
            ("data readiness", "--skip-data-readiness", args.skip_data_readiness),
            (
                "Stage 0 source readiness",
                "--skip-stage0-source-readiness",
                args.skip_stage0_source_readiness,
            ),
            (
                "safety readiness",
                "--skip-safety-readiness",
                args.skip_safety_readiness,
            ),
            ("model readiness", "--skip-model-readiness", args.skip_model_readiness),
        )
        if enabled
    ]


def _diagnostic_nonclaimable_reasons(args: argparse.Namespace) -> list[str]:
    skipped = [label for label, _flag in _skipped_readiness_checks(args)]
    reasons: list[str] = []
    if skipped:
        reasons.append(
            "readiness report skipped preflight checks: " + ", ".join(skipped)
        )
    if args.allow_safety_degraded:
        reasons.append("explicit degraded ADMET/skin-sens evidence was allowed")
    return reasons


def _diagnostic_next_actions(args: argparse.Namespace) -> list[str]:
    actions: list[str] = []
    skipped = _skipped_readiness_checks(args)
    if skipped:
        skipped_flags = ", ".join(flag for _label, flag in skipped)
        actions.append(
            "rerun readiness without diagnostic skip flags before treating this "
            f"run as claimable: remove {skipped_flags}"
        )
    if args.allow_safety_degraded:
        actions.append(
            "rerun readiness and launcher without --allow-safety-degraded before "
            "treating ADMET/skin-sens output as claimable"
        )
    return actions


def _skipped_readiness_group(args: argparse.Namespace) -> dict[str, Any] | None:
    skipped = _skipped_readiness_checks(args)
    if not skipped:
        return None
    return {
        "name": "readiness_skips",
        "kind": "readiness",
        "status": "failed",
        "blocking": True,
        "blockers": [
            {
                "label": label,
                "detail": f"{flag} prevents a ready/claimable readiness result",
            }
            for label, flag in skipped
        ],
    }


def _output_contract(
    args: argparse.Namespace,
    input_type: str | None,
) -> dict[str, Any] | None:
    if args.preset == "stage0":
        return None
    try:
        run_dir = _run_dir(_runner_namespace(args))
    except SystemExit:
        return None
    required_sections = [
        "compound",
        "safety",
        "skin_toxicity",
        "cosmetic_drug",
        "overall_decision",
        "artifacts",
    ]
    if args.preset in {"target-id", "report"}:
        required_sections.extend(["target_prediction", "skin_specialized_binding"])
    verify_cmd = [
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
    ]
    if args.allow_safety_degraded:
        verify_cmd.append("--allow-degraded")
    verified_artifacts = [
        {
            "name": "run_summary_json",
            "path": str(run_dir / "run_summary.json"),
            "recorded_in": "run_verification.json verified_artifacts",
            "checks": ["bytes", "sha256"],
        },
        {
            "name": "run_summary_md",
            "path": str(run_dir / "run_summary.md"),
            "recorded_in": "run_verification.json verified_artifacts",
            "checks": ["bytes", "sha256"],
        },
        {
            "name": "run_verification_log",
            "path": str(run_dir / "run_verification.log"),
            "recorded_in": "run_verification.json verified_artifacts",
            "checks": ["bytes", "sha256"],
        },
    ]
    if args.preset == "report":
        verified_artifacts.append({
            "name": "html_report",
            "path": str(run_dir / "09_report" / "index.html"),
            "recorded_in": "run_verification.json verified_artifacts",
            "checks": ["bytes", "sha256"],
        })
    diagnostic_reasons = _diagnostic_nonclaimable_reasons(args)
    decision_gate_parts = [
        "overall_decision.decision must be PASS for prediction claims",
        "overall_decision.recommended_action must be proceed for prediction claims",
        "overall_decision.requires_human_review must be false for prediction claims",
        "overall_decision.claimable must be true for prediction claims",
    ]
    claimable_when_parts = [
        "workflow succeeds",
        "verify_command exits 0",
        "overall_decision.decision is PASS",
        "overall_decision.recommended_action is proceed",
        "overall_decision.requires_human_review is false",
        "overall_decision.claimable is true",
    ]
    if args.preset in {"target-id", "report"}:
        decision_gate_parts.append("stage0_claim_quality status is ok")
        claimable_when_parts.append("stage0_claim_quality status is ok")
        decision_gate_parts.append(
            "skin_specialized_binding.skin_context_supported is true"
        )
        claimable_when_parts.append(
            "skin_specialized_binding.skin_context_supported is true"
        )
        decision_gate_parts.append(
            "skin_specialized_binding.top_target_skin_context_supported is true"
        )
        claimable_when_parts.append(
            "skin_specialized_binding.top_target_skin_context_supported is true"
        )
    claimable_when_parts.append("diagnostic_nonclaimable_reasons empty")
    decision_gate_parts.append(
        "review_before_claim requires human review and stop_before_claim is not claimable"
    )
    return {
        "run_dir": str(run_dir),
        "summary_json": str(run_dir / "run_summary.json"),
        "summary_md": str(run_dir / "run_summary.md"),
        "verification_json": str(run_dir / "run_verification.json"),
        "verification_log": str(run_dir / "run_verification.log"),
        "html_report": str(run_dir / "09_report" / "index.html")
        if args.preset == "report"
        else None,
        "required_summary_sections": required_sections,
        "required_summary_fields": _summary_field_contract(
            args.preset,
            args.mode,
            input_type,
        ),
        "required_source_artifacts": _source_artifact_contract(args.preset, args.mode),
        "source_evidence_invariants": _source_evidence_invariant_contract(),
        "target_evidence_invariants": _target_evidence_invariant_contract(
            args.preset,
            args.mode,
        ),
        "required_report_markers": _report_marker_contract(
            args.preset,
            input_type,
        ),
        "verified_artifacts": verified_artifacts,
        "required_launcher_result_fields": _launcher_result_field_contract(
            args.preset,
            input_type,
        ),
        "diagnostic_nonclaimable_reasons": diagnostic_reasons,
        "verify_command": shlex.join(verify_cmd),
        "decision_gate": "; ".join(decision_gate_parts),
        "claimable_when": ", ".join(claimable_when_parts),
    }


def _next_actions(groups: list[Mapping[str, Any]], args: argparse.Namespace) -> list[str]:
    actions: list[str] = []
    by_name = {str(group["name"]): group for group in groups}
    source_group = by_name.get("stage0_sources")
    data_group = by_name.get(f"data:{args.preset}")
    claim_quality_group = by_name.get("stage0_claim_quality")
    safety_group = by_name.get("safety")
    model_group = by_name.get("model")
    source_failed = bool(source_group and source_group.get("status") == "failed")

    if source_failed:
        actions.append(
            "python scripts/import_stage0_sources.py --source-dir /path/to/stage0_sources"
        )
    if data_group and data_group.get("status") == "failed":
        data_blockers = _group_blocker_entries(data_group)
        if failed_entries_are_stage0_buildable(data_blockers):
            actions.append(
                "python scripts/run_skinscout.py --preset stage0 --run-id stage0_bootstrap "
                "--cores 16 --allow-stage0-build"
            )
        else:
            actions.append(
                "replace, remove, or force-rebuild invalid/placeholder Stage 0 artifacts; "
                "--allow-stage0-build only bypasses missing generated outputs"
            )
    if data_group and data_group.get("status") == "buildable" and not source_failed:
        actions.append(
            "launch with --allow-stage0-build, or run the Stage 0 bootstrap first"
        )
    if safety_group and safety_group.get("status") == "failed":
        actions.append("python scripts/safety_readiness.py --runtime-mode conda")
    if claim_quality_group and claim_quality_group.get("status") == "failed":
        actions.append("python scripts/stage0_verify.py --strict --claim-quality")
    if model_group and model_group.get("status") == "failed":
        actions.append(
            "micromamba run -n cosmax-boltz2 python scripts/model_readiness.py "
            "--require <missing_requirement>"
        )
    actions.extend(_diagnostic_next_actions(args))
    return list(dict.fromkeys(actions))


def _group_blocker_entries(group: Mapping[str, Any]) -> list[tuple[str, Path]]:
    entries: list[tuple[str, Path]] = []
    blockers = group.get("blockers", [])
    if not isinstance(blockers, list):
        return entries
    for blocker in blockers:
        if not isinstance(blocker, Mapping):
            continue
        label = str(blocker.get("label", ""))
        path = Path(str(blocker.get("path", "")))
        entries.append((label, path))
    return entries


def readiness_payload(args: argparse.Namespace) -> dict[str, Any]:
    run_dti_sanity = not args.no_dti_sanity
    input_group, canonical = _input_group(args)
    raw_input_type = input_group.get("input_type")
    input_type = raw_input_type if isinstance(raw_input_type, str) else None
    groups: list[dict[str, Any]] = [input_group]
    skipped_group = _skipped_readiness_group(args)
    if skipped_group is not None:
        groups.append(skipped_group)

    if _runtime_claims_gpu(args):
        groups.append(_gpu_runtime_group())

    selected_data = None
    if not args.skip_data_readiness:
        selected_data = _data_group(
            name=f"data:{args.preset}",
            preset=args.preset,
            mode=args.mode,
            run_dti_sanity=run_dti_sanity,
            allow_stage0_build=args.allow_stage0_build,
            repo_root=args.repo_root,
        )
        groups.append(selected_data)

    if (
        args.preset in {"target-id", "report"}
        and selected_data is not None
        and selected_data.get("status") == "ok"
    ):
        groups.append(_stage0_claim_quality_group(repo_root=args.repo_root))

    source_required = args.preset == "stage0" or (
        selected_data is not None and selected_data["status"] in {"failed", "buildable"}
    )
    if source_required and not args.skip_stage0_source_readiness:
        groups.append(
            _data_group(
                name="stage0_sources",
                preset="stage0",
                mode=args.mode,
                run_dti_sanity=run_dti_sanity,
                allow_stage0_build=False,
                repo_root=args.repo_root,
            )
        )

    if (
        args.preset in SAFETY_PRESETS
        and not args.skip_safety_readiness
    ):
        groups.append(
            _safety_group(
                runtime_mode=args.runtime_mode,
                online=args.online_safety_readiness,
                allow_degraded=args.allow_safety_degraded,
            )
        )

    if not args.skip_model_readiness:
        model_group = _model_group(
            preset=args.preset,
            mode=args.mode,
            run_dti_sanity=run_dti_sanity,
            use_conda=args.use_conda,
            readiness_command=args.readiness_command,
        )
        if model_group["required"]:
            groups.append(model_group)

    run_manifest_v2: dict[str, object] | None = None
    if args.evidence_mode == "discovery":
        alias_dir = args.discovery_alias_dir
        if not alias_dir.is_absolute():
            alias_dir = args.repo_root / alias_dir
        groups.append(_discovery_alias_group(alias_dir, args.repo_root))
    if not input_group.get("blocking"):
        try:
            run_manifest_v2 = _run_manifest_payload(_runner_namespace(args))
        except (OSError, SystemExit, ValueError) as exc:
            groups.append({
                "name": "run-contract-v2",
                "kind": "contract",
                "status": "failed",
                "blocking": True,
                "blockers": [{
                    "label": "run_manifest_v2",
                    "path": "schemas/run_manifest_v2.json",
                    "detail": str(exc),
                }],
            })
            run_manifest_v2 = None
    blocking = [group for group in groups if group.get("blocking")]
    legacy_profile = args.run_profile.legacy_fields()
    skipped_blocked = any(group.get("name") == "readiness_skips" for group in blocking)
    only_skipped_blocked = skipped_blocked and all(
        group.get("name") == "readiness_skips" for group in blocking
    )
    payload = {
        "schema_version": 1,
        "status": "ok" if not blocking else "readiness_blocked" if only_skipped_blocked else "failed",
        **legacy_profile,
        "run_profile": args.run_profile.to_dict(),
        "run_manifest_v2": run_manifest_v2,
        "run_id": input_group.get("run_id"),
        "input_type": input_type,
        "input_smiles": input_group.get("input_smiles"),
        "input_canonical_smiles": input_group.get("input_canonical_smiles"),
        "input_sdf": input_group.get("input_sdf"),
        "canonical_smiles": canonical,
        "groups": groups,
        "next_actions": _next_actions(groups, args),
        "snakemake_command": _snakemake_command(args),
        "output_contract": (
            None
            if input_group.get("blocking")
            else _output_contract(args, input_type)
        ),
    }
    return payload


def _print_text(payload: Mapping[str, Any]) -> None:
    print(f"SkinScout pipeline readiness: {payload['status']}")
    print(f"preset={payload['preset']} mode={payload['mode']} run_id={payload['run_id']}")
    if payload.get("input_type"):
        print(f"input_type={payload['input_type']}")
    if payload.get("input_smiles"):
        print(f"input_smiles={payload['input_smiles']}")
    if payload.get("input_canonical_smiles"):
        print(f"input_canonical_smiles={payload['input_canonical_smiles']}")
    if payload.get("input_sdf"):
        print(f"input_sdf={payload['input_sdf']}")
    if payload.get("canonical_smiles"):
        print(f"canonical_smiles={payload['canonical_smiles']}")
    output_contract = payload.get("output_contract")
    if isinstance(output_contract, Mapping):
        print(f"run_dir={output_contract['run_dir']}")
    print("")
    for group in payload.get("groups", []):
        if not isinstance(group, Mapping):
            continue
        blockers = group.get("blockers", [])
        n_blockers = len(blockers) if isinstance(blockers, list) else 0
        print(f"[{group['status']}] {group['name']} ({n_blockers} blocker(s))")
        if not isinstance(blockers, list):
            continue
        for blocker in blockers[:8]:
            if not isinstance(blocker, Mapping):
                continue
            label = blocker.get("label", "unknown")
            path = blocker.get("path")
            detail = blocker.get("detail")
            suffix = f": {path}" if path else f": {detail}" if detail else ""
            print(f"  - {label}{suffix}")
        if n_blockers > 8:
            print(f"  - ... {n_blockers - 8} more")
    actions = payload.get("next_actions", [])
    if actions:
        print("\nNext actions:")
        for action in actions:
            print(f"  - {action}")
    command = payload.get("snakemake_command")
    if command:
        print("\nSnakemake command:")
        print(f"  {command}")
    if isinstance(output_contract, Mapping):
        print("\nExpected completed-run contract:")
        print(f"  summary_json: {output_contract['summary_json']}")
        print(f"  summary_md: {output_contract['summary_md']}")
        verification_json = output_contract.get("verification_json")
        if verification_json:
            print(f"  verification_json: {verification_json}")
        verification_log = output_contract.get("verification_log")
        if verification_log:
            print(f"  verification_log: {verification_log}")
        html_report = output_contract.get("html_report")
        if html_report:
            print(f"  html_report: {html_report}")
        print(f"  verify: {output_contract['verify_command']}")
        decision_gate = output_contract.get("decision_gate")
        if decision_gate:
            print(f"  decision gate: {decision_gate}")
        claimable_when = output_contract.get("claimable_when")
        if claimable_when:
            print(f"  claimable when: {claimable_when}")
        diagnostic_reasons = output_contract.get("diagnostic_nonclaimable_reasons", [])
        if isinstance(diagnostic_reasons, list) and diagnostic_reasons:
            print("  diagnostic non-claimable reasons:")
            for reason in diagnostic_reasons:
                print(f"    - {reason}")
        sections = output_contract.get("required_summary_sections", [])
        if isinstance(sections, list):
            print("  required sections: " + ", ".join(str(item) for item in sections))
        fields = output_contract.get("required_summary_fields", {})
        if isinstance(fields, Mapping):
            for section, names in fields.items():
                if isinstance(names, list):
                    print(
                        f"  required {section} fields: "
                        + ", ".join(str(item) for item in names)
                    )
        artifacts = output_contract.get("required_source_artifacts", [])
        if isinstance(artifacts, list) and artifacts:
            print("  required source artifacts: " + ", ".join(str(item) for item in artifacts))
        verified_artifacts = output_contract.get("verified_artifacts", [])
        if isinstance(verified_artifacts, list) and verified_artifacts:
            print("  verified artifacts:")
            for artifact in verified_artifacts:
                if not isinstance(artifact, Mapping):
                    continue
                checks = artifact.get("checks", [])
                checks_text = (
                    ", ".join(str(check) for check in checks)
                    if isinstance(checks, list)
                    else str(checks)
                )
                print(
                    "    - "
                    f"{artifact.get('name')}: {artifact.get('path')} "
                    f"({checks_text})"
                )
        invariants = output_contract.get("source_evidence_invariants", [])
        if isinstance(invariants, list) and invariants:
            print("  source evidence invariants:")
            for invariant in invariants:
                print(f"    - {invariant}")
        target_invariants = output_contract.get("target_evidence_invariants", [])
        if isinstance(target_invariants, list) and target_invariants:
            print("  target evidence invariants:")
            for invariant in target_invariants:
                print(f"    - {invariant}")
        markers = output_contract.get("required_report_markers", [])
        if isinstance(markers, list) and markers:
            print(
                "  required report markers: "
                + ", ".join(str(item) for item in markers)
            )
        launcher_fields = output_contract.get("required_launcher_result_fields", [])
        if isinstance(launcher_fields, list) and launcher_fields:
            print(
                "  required launcher result fields: "
                + ", ".join(str(item) for item in launcher_fields)
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compound", nargs="?", help="Input compound SMILES")
    parser.add_argument("--smiles", help="Input compound SMILES")
    parser.add_argument("--sdf", type=Path, help="Input compound SDF path")
    parser.add_argument(
        "--run-id",
        help="Safe output run identifier; defaults to the same deterministic ID as run_skinscout.py",
    )
    parser.add_argument("--mode", choices=sorted(MODES), default="comprehensive")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="target-id")
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
        help="Evidence is the default; Discovery requires exact-record exclusion.",
    )
    parser.add_argument("--cores", type=int, default=16)
    parser.add_argument("--resources", action="append", default=["gpu=1"])
    parser.add_argument("--snakemake", default="snakemake")
    parser.add_argument("--no-use-conda", dest="use_conda", action="store_false")
    parser.set_defaults(use_conda=True)
    parser.add_argument(
        "--conda-frontend",
        choices=["auto", "conda", "mamba"],
        default="auto",
    )
    parser.add_argument("--no-dti-sanity", action="store_true")
    parser.add_argument("--extra-config", action="append", default=[])
    parser.add_argument(
        "--discovery-alias-dir",
        default=Path("data/discovery_aliases"),
        type=Path,
        help="Canonical Stage 0 exact-alias package used for Discovery exclusion.",
    )
    parser.add_argument(
        "--target-metadata",
        type=Path,
        default=DEFAULT_TARGET_METADATA,
        help="HPA-style TSV mapping UniProt IDs to gene/protein labels.",
    )
    parser.add_argument("--allow-safety-degraded", action="store_true")
    parser.add_argument("--online-safety-readiness", action="store_true")
    parser.add_argument("--allow-stage0-build", action="store_true")
    parser.add_argument(
        "--runtime-mode",
        choices=["conda", "active", "gpu"],
        default="conda",
        help="Safety readiness runtime mode; gpu additionally hard-gates host CUDA.",
    )
    parser.add_argument("--skip-data-readiness", action="store_true")
    parser.add_argument("--skip-stage0-source-readiness", action="store_true")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=ROOT,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--skip-safety-readiness", action="store_true")
    parser.add_argument("--skip-model-readiness", action="store_true")
    parser.add_argument(
        "--readiness-command",
        help=(
            "Command prefix for model readiness, default: "
            "micromamba run -n cosmax-boltz2 python"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    if args.compound:
        if args.smiles or args.sdf:
            parser.error("positional SMILES cannot be combined with --smiles or --sdf")
        args.smiles = args.compound
    _normalize_preset_args(args)
    if args.sota_claim and args.preset not in {"target-id", "report"}:
        parser.error("--sota-claim is only valid with target-id/report SOTA runs")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = readiness_payload(args)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.json_out.with_suffix(args.json_out.suffix + ".tmp")
        tmp.write_text(text)
        tmp.replace(args.json_out)
    if args.json:
        sys.stdout.write(text)
    else:
        _print_text(payload)
    return 0 if payload["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
