#!/usr/bin/env python3
"""Report model/tool readiness for SkinScout GPU scoring stages."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import gpu_admission
from boltz2_runner import (
    affinity_score,
    load_affinity_payload,
    nonempty,
    run_boltz_predict,
    write_affinity_yaml,
)


def _add_repo_root_to_path() -> None:
    root = Path(__file__).resolve().parents[1]
    root_text = str(root)
    if root_text in sys.path:
        return
    pythonpath_entries = [
        path for path in os.environ.get("PYTHONPATH", "").split(os.pathsep) if path
    ]
    sys.path.insert(min(len(sys.path), 1 + len(pythonpath_entries)), root_text)


_add_repo_root_to_path()

ROOT = Path(__file__).resolve().parents[1]
LOCAL_COMMAND_CANDIDATES = {
    "gnina": [ROOT / "tools/gnina"],
    "autodock_gpu_128wi": [
        ROOT / "tools/autodock_gpu/bin/autodock_gpu_128wi",
        ROOT / "tools/autodock_gpu/autodock_gpu_128wi",
        ROOT / "tools/AutoDock-GPU/bin/autodock_gpu_128wi",
    ],
    "autogrid4": [
        ROOT / "tools/autogrid/bin/autogrid4",
        ROOT / "tools/AutoGrid/autogrid4",
        ROOT / "tools/AutoGrid/bin/autogrid4",
    ],
}
PACKAGE_ALIASES = {
    "autodock_gpu": {"autodock-gpu"},
    "autodock-gpu": {"autodock_gpu"},
    "gmx_mmpbsa": {"gmx-mmpbsa"},
    "gmx-mmpbsa": {"gmx_mmpbsa"},
    "scikit_learn": {"scikit-learn"},
    "scikit-learn": {"scikit_learn"},
}
# Requirements evaluated directly against the payload instead of through
# STATUS_REQUIREMENTS' (section, name, expected) lookup.
SPECIAL_REQUIREMENTS = frozenset({"boltz_smoke", "gpu", "gpu_execution"})

STATUS_REQUIREMENTS = {
    "boltz": ("tools", "boltz", "available"),
    "gnina": ("tools", "gnina", "available"),
    "autodock_gpu": ("tools", "autodock_gpu", "available"),
    "autogrid": ("tools", "autogrid", "available"),
    "meeko": ("tools", "meeko", "available"),
    "bioemu": ("tools", "bioemu", "available"),
    "gromacs": ("tools", "gromacs", "available"),
    "gmx_mmpbsa": ("tools", "gmx_mmpbsa", "available"),
    "acpype": ("tools", "acpype", "available"),
    "xtb": ("tools", "xtb", "available"),
    "crest": ("tools", "crest", "available"),
    "obabel": ("tools", "obabel", "available"),
    # Mandatory for the comprehensive DAG: autodock_pick_top_pct takes
    # diffdock_blind_no_pocket's output as a required input, and nothing in the
    # installers provides it, so it went unchecked entirely.
    "diffdock": ("tools", "diffdock", "available"),
    "rtmscore": ("imports", "rtmscore", "available"),
    "psichic": ("imports", "psichic", "available"),
    "mdtraj": ("imports", "mdtraj", "available"),
    "sklearn": ("imports", "sklearn", "available"),
    "pyscf": ("imports", "pyscf", "available"),
    "gpu4pyscf": ("imports", "gpu4pyscf", "available"),
    "autodock_gpu_env": ("envs", "autodock_gpu", "declared"),
    "meeko_env": ("envs", "meeko", "declared"),
    "bioemu_env": ("envs", "bioemu", "declared"),
    "md_env": ("envs", "md", "declared"),
    "qm_env": ("envs", "qm", "declared"),
}


def autogrid_status(
    *,
    discover_inactive_conda_envs: bool = True,
) -> dict[str, object]:
    status = (
        command_status("autogrid4")
        if discover_inactive_conda_envs
        else command_status("autogrid4", discover_inactive_conda_envs=False)
    )
    primary_parameter_candidates = [
        Path(os.environ["AUTODOCK_PARAMETER_FILE"])
        if os.environ.get("AUTODOCK_PARAMETER_FILE")
        else None,
        ROOT / "tools/AutoDock-GPU/AD4_parameters.dat",
    ]
    bound_parameter_candidates = [
        ROOT / "tools/AutoGrid/AD4.1_bound.dat",
        ROOT / "tools/autogrid/share/AD4.1_bound.dat",
        ROOT / "tools/AutoGrid/parameter_library/AD4.1_bound.dat",
    ]
    primary_parameter = next(
        (
            candidate
            for candidate in primary_parameter_candidates
            if candidate is not None
            and candidate.is_file()
            and candidate.stat().st_size > 0
        ),
        None,
    )
    bound_parameter = next(
        (
            candidate
            for candidate in bound_parameter_candidates
            if candidate.is_file() and candidate.stat().st_size > 0
        ),
        None,
    )
    status["parameter_file"] = (
        str(primary_parameter) if primary_parameter is not None else None
    )
    status["bound_parameter_file"] = (
        str(bound_parameter) if bound_parameter is not None else None
    )
    # The pipeline resolves one parameter file from a candidate list
    # (stage3_autogrid_maps._resolve_parameter_file), and AD4.1_bound.dat is one
    # of those alternatives rather than a second requirement. Demanding both
    # marked a working install unusable and blocked the browser from starting
    # Target analyses - the proteome run that produced 13,339 maps used
    # AD4_parameters.dat alone. Requiring the primary specifically had the
    # mirror-image effect: a bound-only install that the pipeline accepts was
    # still reported missing.
    resolved = primary_parameter if primary_parameter is not None else bound_parameter
    status["resolved_parameter_file"] = str(resolved) if resolved is not None else None
    if status.get("status") == "available" and resolved is None:
        status["status"] = "missing"
        status["reason"] = "AutoDock4 parameter file is not available"
    return status


def _conda_env_roots() -> list[Path]:
    """Environment prefixes Snakemake activates per rule.

    Each workflow rule declares its own env, so tools like autogrid4, meeko,
    xtb, obabel and boltz are never on the caller's PATH. Reporting them
    missing blocked the browser from starting analyses whose tools were in
    fact installed.
    """
    prefix = os.environ.get("MAMBA_ROOT_PREFIX") or os.environ.get("CONDA_PREFIX")
    roots: list[Path] = []
    for base in (
        Path(prefix).parent if prefix and Path(prefix).name != "envs" else None,
        Path(prefix) if prefix else None,
        Path.home() / ".local" / "share" / "mamba",
        Path.home() / "micromamba",
        Path.home() / "miniforge3",
        Path.home() / "miniconda3",
    ):
        if base is None:
            continue
        envs = base / "envs"
        if envs.is_dir() and envs not in roots:
            roots.append(envs)
    return roots


def _command_in_conda_envs(name: str) -> str | None:
    for envs in _conda_env_roots():
        for candidate in sorted(envs.glob(f"*/bin/{name}")):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
    return None


def command_status(
    name: str,
    *,
    discover_inactive_conda_envs: bool = True,
) -> dict[str, object]:
    path = shutil.which(name)
    if path is not None:
        return {"status": "available", "path": path}
    if os.environ.get("SKINSCOUT_DISABLE_LOCAL_TOOL_PATHS") == "1":
        return {"status": "missing", "path": None}
    for candidate in LOCAL_COMMAND_CANDIDATES.get(name, []):
        if candidate.exists() and os.access(candidate, os.X_OK):
            return {"status": "available", "path": str(candidate)}
    if discover_inactive_conda_envs:
        env_path = _command_in_conda_envs(name)
        if env_path is not None:
            return {"status": "available", "path": env_path, "source": "conda_env"}
    return {"status": "missing", "path": None}


def _package_name(spec: str) -> str:
    return re.split(r"[=<>!~\s]", spec.strip(), maxsplit=1)[0].lower()


def _package_aliases(spec: str) -> set[str]:
    normalized = _package_name(spec)
    return {normalized, *PACKAGE_ALIASES.get(normalized, set())}


def _env_packages(env_yml: Path) -> tuple[set[str], set[str]]:
    import yaml

    payload = yaml.safe_load(env_yml.read_text())
    if not isinstance(payload, dict):
        return set(), set()
    dependencies = payload.get("dependencies")
    if not isinstance(dependencies, list):
        return set(), set()

    conda: set[str] = set()
    pip: set[str] = set()
    for item in dependencies:
        if isinstance(item, str):
            conda.add(_package_name(item))
        elif isinstance(item, dict):
            pip_items = item.get("pip")
            if isinstance(pip_items, list):
                pip.update(_package_name(str(dep)) for dep in pip_items)
    return conda, pip


def env_package_status(
    env_yml: Path,
    package: str,
    *,
    label: str | None = None,
) -> dict[str, object]:
    status = env_packages_status(env_yml, [package], label=label or package)
    status["package"] = package
    return status


def env_packages_status(
    env_yml: Path,
    packages: list[str],
    *,
    label: str,
) -> dict[str, object]:
    if not env_yml.exists():
        return {
            "status": "missing",
            "packages": packages,
            "label": label,
            "env_yml": str(env_yml),
            "reason": "missing environment manifest",
        }
    conda, pip = _env_packages(env_yml)
    declared = conda | pip
    missing = [
        package
        for package in packages
        if not (_package_aliases(package) & declared)
    ]
    return {
        "status": "declared" if not missing else "missing",
        "packages": packages,
        "label": label,
        "env_yml": str(env_yml),
        "missing": missing,
        "reason": None if not missing else "packages not declared in environment manifest",
    }


def import_status(module: str) -> dict[str, object]:
    spec = importlib.util.find_spec(module)
    if spec is None:
        return {"status": "missing", "module": module}
    return {"status": "available", "module": module}


def diffdock_status(
    *,
    discover_inactive_conda_envs: bool = True,
) -> dict[str, object]:
    """A `diffdock` command taking the flags stage3_diffdock_blind.py passes.

    Upstream ships inference.py rather than a console command, so this is
    normally a wrapper on PATH in the same style as the GNINA one.
    """
    status = (
        command_status("diffdock")
        if discover_inactive_conda_envs
        else command_status("diffdock", discover_inactive_conda_envs=False)
    )
    status["required_flags"] = [
        "--protein_path",
        "--ligand",
        "--out_dir",
        "--samples_per_complex",
        "--inference_steps",
    ]
    status["expected_output"] = "rank1_confidence*.sdf in --out_dir"
    if status["status"] == "missing":
        status["reason"] = (
            "comprehensive mode cannot start without it; see README "
            "'Comprehensive 모드가 추가로 요구하는 것'"
        )
    return status


def rtmscore_status() -> dict[str, object]:
    status = import_status("rtmscore")
    root = Path(
        os.environ.get(
            "RTMSCORE_ROOT",
            str(Path.home() / ".local" / "opt" / "RTMScore"),
        )
    )
    model = root / "trained_models" / "rtmscore_model1.pth"
    script = root / "example" / "rtmscore.py"
    status.update({"root": str(root), "model": str(model)})
    if status["status"] == "missing":
        return status
    missing = [str(path) for path in (script, model) if not path.exists()]
    if missing:
        status.update(
            {
                "status": "missing",
                "reason": "missing upstream RTMScore files",
                "missing": missing,
            }
        )
    return status


def psichic_status() -> dict[str, object]:
    status = import_status("psichic")
    root = Path(
        os.environ.get(
            "PSICHIC_ROOT",
            str(Path.home() / ".local" / "opt" / "PSICHIC"),
        )
    )
    model_name = os.environ.get("PSICHIC_MODEL", "PDBv2020_PSICHIC")
    inference = root / "PSICHIC-prod" / "inference.py"
    model = root / "trained_weights" / model_name / "model.pt"
    status.update(
        {"root": str(root), "model": str(model), "model_name": model_name}
    )
    if status["status"] == "missing":
        return status
    missing = [str(path) for path in (inference, model) if not path.exists()]
    if missing:
        status.update(
            {
                "status": "missing",
                "reason": "missing upstream PSICHIC files",
                "missing": missing,
            }
        )
    return status


def cuda_status() -> dict[str, object]:
    spec = importlib.util.find_spec("torch")
    if spec is None:
        execution = gpu_admission.probe(cwd=ROOT)
        detected = execution.get("detected") is True
        device_details = execution.get("device_details")
        first_device = (
            device_details[0]
            if isinstance(device_details, list)
            and device_details
            and isinstance(device_details[0], dict)
            else {}
        )
        return {
            "status": "available" if detected else "missing",
            "backend": "nvidia-smi",
            "cuda_available": detected,
            "device_count": len(device_details)
            if detected and isinstance(device_details, list)
            else 0,
            "device_name": first_device.get("name") if detected else None,
            "torch_available": False,
        }
    import torch  # type: ignore[import-not-found]

    return {
        "status": "available",
        "backend": "pytorch",
        "version": getattr(torch, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }


def boltz_smoke(receptor_pdb: Path, ligand_sdf: Path) -> dict[str, object]:
    if not nonempty(receptor_pdb):
        return {"status": "blocked", "reason": f"missing receptor: {receptor_pdb}"}
    if not nonempty(ligand_sdf):
        return {"status": "blocked", "reason": f"missing ligand: {ligand_sdf}"}
    if shutil.which("boltz") is None:
        return {"status": "missing", "reason": "boltz not on PATH"}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        input_yaml = tmp_dir / "input.yaml"
        out_dir = tmp_dir / "boltz_out"
        try:
            write_affinity_yaml(receptor_pdb, ligand_sdf, input_yaml)
        except ValueError as exc:
            return {"status": "blocked", "reason": str(exc)}
        res = run_boltz_predict(input_yaml, out_dir)
        if res is None:
            return {"status": "missing", "reason": "boltz not on PATH"}
        if res.returncode != 0:
            return {
                "status": "failed",
                "returncode": res.returncode,
                "stderr_tail": res.stderr[-1200:],
            }
        payload = load_affinity_payload(out_dir)
        score = affinity_score(payload) if payload is not None else None
        if score is None:
            return {"status": "failed", "reason": "no usable affinity output"}
        return {"status": "passed", "score": score}


def readiness(args: argparse.Namespace) -> dict[str, object]:
    autodock_env = ROOT / "envs/autodock_gpu.yml"
    meeko_env = ROOT / "envs/meeko.yml"
    bioemu_env = ROOT / "envs/bioemu.yml"
    md_env = ROOT / "envs/md.yml"
    qm_env = ROOT / "envs/qm.yml"
    discover_inactive = not args.active_environment_only

    def tool(name: str) -> dict[str, object]:
        return command_status(
            name,
            discover_inactive_conda_envs=discover_inactive,
        )

    payload: dict[str, object] = {
        "tools": {
            "boltz": tool("boltz"),
            "gnina": tool("gnina"),
            "autodock_gpu": tool("autodock_gpu_128wi"),
            "autogrid": autogrid_status(
                discover_inactive_conda_envs=discover_inactive,
            ),
            "meeko": tool("mk_prepare_ligand.py"),
            "bioemu": tool("bioemu"),
            "gromacs": tool("gmx"),
            "gmx_mmpbsa": tool("gmx_MMPBSA"),
            "acpype": tool("acpype"),
            "xtb": tool("xtb"),
            "crest": tool("crest"),
            "obabel": tool("obabel"),
            "diffdock": diffdock_status(
                discover_inactive_conda_envs=discover_inactive,
            ),
        },
        "imports": {
            "rtmscore": rtmscore_status(),
            "psichic": psichic_status(),
            "mdtraj": import_status("mdtraj"),
            "sklearn": import_status("sklearn"),
            "pyscf": import_status("pyscf"),
            "gpu4pyscf": import_status("gpu4pyscf"),
        },
        "envs": {
            "autodock_gpu": env_package_status(
                autodock_env,
                "vina",
                label="AutoDock support Snakemake env",
            ),
            "meeko": env_package_status(
                meeko_env,
                "meeko",
                label="Meeko ligand prep Snakemake env",
            ),
            "bioemu": env_packages_status(
                bioemu_env,
                ["bioemu", "mdtraj", "scikit-learn"],
                label="BioEmu Snakemake env",
            ),
            "md": env_packages_status(
                md_env,
                ["gromacs", "acpype", "gmx-mmpbsa"],
                label="GROMACS/MM-GBSA Snakemake env",
            ),
            "qm": env_packages_status(
                qm_env,
                ["xtb", "crest", "pyscf", "openbabel"],
                label="QM/CREST Snakemake env",
            ),
        },
        "gpu": cuda_status(),
        "gpu_execution": gpu_admission.probe(cwd=ROOT),
    }
    if args.boltz_smoke_receptor is not None or args.boltz_smoke_ligand is not None:
        if args.boltz_smoke_receptor is None or args.boltz_smoke_ligand is None:
            payload["boltz_smoke"] = {
                "status": "blocked",
                "reason": "both --boltz-smoke-receptor and --boltz-smoke-ligand are required",
            }
        else:
            payload["boltz_smoke"] = boltz_smoke(
                args.boltz_smoke_receptor,
                args.boltz_smoke_ligand,
            )
    return payload


def _has_required_failures(payload: dict[str, object], required: list[str]) -> bool:
    return bool(failed_requirements(payload, required))


def failed_requirements(
    payload: dict[str, object],
    required: list[str],
) -> list[dict[str, object]]:
    failures: list[dict[str, object]] = []
    gpu = payload.get("gpu", {})
    gpu_execution = payload.get("gpu_execution", {})
    smoke = payload.get("boltz_smoke", {})
    for requirement in required:
        if requirement == "gpu":
            if isinstance(gpu, dict) and gpu.get("cuda_available") is True:
                continue
            failures.append(
                {
                    "requirement": requirement,
                    "section": "gpu",
                    "expected": "cuda_available=true",
                    "actual": gpu,
                }
            )
            continue
        if requirement == "gpu_execution":
            if (
                isinstance(gpu_execution, dict)
                and gpu_execution.get("available_for_analysis") is True
            ):
                continue
            failures.append(
                {
                    "requirement": requirement,
                    "section": "gpu_execution",
                    "expected": (
                        f"available_for_analysis=true with at least "
                        f"{gpu_admission.MIN_FREE_MIB} MiB free"
                    ),
                    "actual": gpu_execution,
                }
            )
            continue
        if requirement == "boltz_smoke":
            if isinstance(smoke, dict) and smoke.get("status") == "passed":
                continue
            failures.append(
                {
                    "requirement": requirement,
                    "section": "boltz_smoke",
                    "expected": "passed",
                    "actual": smoke,
                }
            )
            continue
        check = STATUS_REQUIREMENTS.get(requirement)
        if check is None:
            failures.append(
                {
                    "requirement": requirement,
                    "section": "unknown",
                    "expected": "registered readiness requirement",
                    "actual": None,
                }
            )
            continue
        section, name, expected = check
        section_payload = payload.get(section, {})
        item = (
            section_payload.get(name, {})
            if isinstance(section_payload, dict)
            else {}
        )
        if isinstance(item, dict) and item.get("status") == expected:
            continue
        failures.append(
            {
                "requirement": requirement,
                "section": section,
                "name": name,
                "expected": expected,
                "actual": item,
            }
        )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--boltz-smoke-receptor", type=Path)
    parser.add_argument("--boltz-smoke-ligand", type=Path)
    parser.add_argument(
        "--active-environment-only",
        action="store_true",
        help="Do not count executables found only in inactive Conda environments.",
    )
    parser.add_argument(
        "--require",
        action="append",
        # Derived from the requirement tables rather than restated, so a check
        # can never be registered yet be unselectable. `diffdock` was in
        # STATUS_REQUIREMENTS but missing from a hand-maintained list here, so
        # `--require diffdock` was rejected by argparse before it could run -
        # and DiffDock is a hard input of the comprehensive DAG.
        choices=sorted(set(STATUS_REQUIREMENTS) | SPECIAL_REQUIREMENTS),
        default=[],
        help="Exit non-zero if the named readiness check is not available/passed.",
    )
    args = parser.parse_args()

    payload = readiness(args)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.json_out.with_suffix(args.json_out.suffix + ".tmp")
        tmp.write_text(text)
        tmp.replace(args.json_out)
    sys.stdout.write(text)
    if _has_required_failures(payload, args.require):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
