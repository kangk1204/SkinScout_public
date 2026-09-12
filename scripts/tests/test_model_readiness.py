"""Regression tests for model readiness diagnostics."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
READINESS = ROOT / "scripts/model_readiness.py"


def load_model_readiness_module():
    scripts_dir = str(ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "model_readiness_under_test",
        READINESS,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_model_readiness_reports_missing_tools_with_empty_path(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PATH"] = str(tmp_path / "empty")
    env["SKINSCOUT_DISABLE_LOCAL_TOOL_PATHS"] = "1"
    Path(env["PATH"]).mkdir()

    res = subprocess.run(
        [sys.executable, str(ROOT / "scripts/model_readiness.py")],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode == 0
    payload = json.loads(res.stdout)
    assert payload["tools"]["boltz"]["status"] == "missing"
    assert payload["tools"]["gnina"]["status"] == "missing"
    assert payload["tools"]["autodock_gpu"]["status"] == "missing"
    assert payload["tools"]["meeko"]["status"] == "missing"
    assert payload["tools"]["bioemu"]["status"] == "missing"
    assert payload["tools"]["gromacs"]["status"] == "missing"
    assert payload["tools"]["gmx_mmpbsa"]["status"] == "missing"
    assert payload["tools"]["acpype"]["status"] == "missing"
    assert payload["tools"]["xtb"]["status"] == "missing"
    assert payload["tools"]["crest"]["status"] == "missing"
    assert payload["tools"]["obabel"]["status"] == "missing"
    assert payload["envs"]["autodock_gpu"]["status"] == "declared"
    assert payload["envs"]["meeko"]["status"] == "declared"
    assert payload["envs"]["bioemu"]["status"] == "declared"
    assert payload["envs"]["md"]["status"] == "declared"
    assert payload["envs"]["qm"]["status"] == "declared"
    assert "gpu" in payload


def test_model_readiness_require_boltz_exits_nonzero_when_missing(
    tmp_path: Path,
) -> None:
    env = os.environ.copy()
    env["PATH"] = str(tmp_path / "empty")
    env["SKINSCOUT_DISABLE_LOCAL_TOOL_PATHS"] = "1"
    Path(env["PATH"]).mkdir()

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/model_readiness.py"),
            "--require",
            "boltz",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode == 2
    payload = json.loads(res.stdout)
    assert payload["tools"]["boltz"]["status"] == "missing"


def test_model_readiness_require_active_autodock_exits_nonzero_when_missing(
    tmp_path: Path,
) -> None:
    env = os.environ.copy()
    env["PATH"] = str(tmp_path / "empty")
    env["SKINSCOUT_DISABLE_LOCAL_TOOL_PATHS"] = "1"
    Path(env["PATH"]).mkdir()

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/model_readiness.py"),
            "--require",
            "autodock_gpu",
            "--require",
            "meeko",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode == 2
    payload = json.loads(res.stdout)
    assert payload["tools"]["autodock_gpu"]["status"] == "missing"
    assert payload["tools"]["meeko"]["status"] == "missing"


def test_model_readiness_require_snakemake_autodock_env_passes() -> None:
    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/model_readiness.py"),
            "--require",
            "autodock_gpu_env",
            "--require",
            "meeko_env",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode == 0, res.stderr
    payload = json.loads(res.stdout)
    assert payload["envs"]["autodock_gpu"]["status"] == "declared"
    assert payload["envs"]["meeko"]["status"] == "declared"


def test_autogrid_readiness_needs_the_parameter_file_it_actually_uses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pipeline resolves one parameter file from a candidate list.

    AD4.1_bound.dat is one of those alternatives, not a second requirement.
    Requiring both reported a working install as unusable and, because
    analysis readiness gates on that, stopped the browser from starting Target
    analyses - while the proteome run built 13,339 maps from AD4_parameters.dat
    alone.
    """
    readiness = load_model_readiness_module()
    monkeypatch.setattr(readiness, "ROOT", tmp_path)
    monkeypatch.setattr(
        readiness,
        "command_status",
        lambda _name: {"status": "available", "path": "/fake/autogrid4"},
    )

    without_any = readiness.autogrid_status()
    assert without_any["status"] == "missing"
    assert without_any["parameter_file"] is None

    primary = tmp_path / "tools/AutoDock-GPU/AD4_parameters.dat"
    primary.parent.mkdir(parents=True)
    primary.write_text("primary\n", encoding="utf-8")

    usable = readiness.autogrid_status()
    assert usable["status"] == "available"
    assert usable["parameter_file"] == str(primary)
    # Still reported so an operator can see which libraries are present.
    assert usable["bound_parameter_file"] is None

    bound = tmp_path / "tools/AutoGrid/AD4.1_bound.dat"
    bound.parent.mkdir(parents=True)
    bound.write_text("bound\n", encoding="utf-8")
    both = readiness.autogrid_status()
    assert both["bound_parameter_file"] == str(bound)
    assert both["resolved_parameter_file"] == str(primary)

    # A bound-only install is one the pipeline accepts, so readiness must too.
    primary.unlink()
    bound_only = readiness.autogrid_status()
    assert bound_only["status"] == "available"
    assert bound_only["parameter_file"] is None
    assert bound_only["resolved_parameter_file"] == str(bound)


def test_model_readiness_require_snakemake_downstream_envs_passes() -> None:
    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/model_readiness.py"),
            "--require",
            "bioemu_env",
            "--require",
            "md_env",
            "--require",
            "qm_env",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode == 0, res.stderr
    payload = json.loads(res.stdout)
    assert payload["envs"]["bioemu"]["status"] == "declared"
    assert payload["envs"]["md"]["status"] == "declared"
    assert payload["envs"]["qm"]["status"] == "declared"
    assert payload["envs"]["qm"]["missing"] == []
    assert "openbabel" in payload["envs"]["qm"]["packages"]


def test_model_readiness_require_active_downstream_tools_exits_nonzero_when_missing(
    tmp_path: Path,
) -> None:
    env = os.environ.copy()
    env["PATH"] = str(tmp_path / "empty")
    Path(env["PATH"]).mkdir()
    # Emptying PATH is no longer enough: tools also resolve through the conda
    # environments Snakemake activates per rule, so the isolation flag is what
    # makes "not installed" mean it.
    env["SKINSCOUT_DISABLE_LOCAL_TOOL_PATHS"] = "1"

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/model_readiness.py"),
            "--require",
            "bioemu",
            "--require",
            "gromacs",
            "--require",
            "gmx_mmpbsa",
            "--require",
            "acpype",
            "--require",
            "xtb",
            "--require",
            "crest",
            "--require",
            "obabel",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode == 2
    payload = json.loads(res.stdout)
    assert payload["tools"]["bioemu"]["status"] == "missing"
    assert payload["tools"]["gromacs"]["status"] == "missing"
    assert payload["tools"]["gmx_mmpbsa"]["status"] == "missing"
    assert payload["tools"]["acpype"]["status"] == "missing"
    assert payload["tools"]["xtb"]["status"] == "missing"
    assert payload["tools"]["crest"]["status"] == "missing"
    assert payload["tools"]["obabel"]["status"] == "missing"


def test_model_readiness_writes_json_out(tmp_path: Path) -> None:
    out = tmp_path / "readiness.json"

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/model_readiness.py"),
            "--json-out",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode == 0
    assert json.loads(out.read_text()) == json.loads(res.stdout)


def test_model_readiness_reports_missing_adapter_roots(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["RTMSCORE_ROOT"] = str(tmp_path / "missing_rtm")
    env["PSICHIC_ROOT"] = str(tmp_path / "missing_psichic")

    res = subprocess.run(
        [sys.executable, str(ROOT / "scripts/model_readiness.py")],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode == 0
    payload = json.loads(res.stdout)
    assert payload["imports"]["rtmscore"]["status"] == "missing"
    assert payload["imports"]["rtmscore"]["reason"] == "missing upstream RTMScore files"
    assert payload["imports"]["psichic"]["status"] == "missing"
    assert payload["imports"]["psichic"]["reason"] == "missing upstream PSICHIC files"


def test_cuda_status_uses_driver_probe_when_demo_runtime_has_no_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness = load_model_readiness_module()
    monkeypatch.setattr(readiness.importlib.util, "find_spec", lambda _name: None)
    monkeypatch.setattr(
        readiness.gpu_admission,
        "probe",
        lambda **_kwargs: {
            "detected": True,
            "device_details": [{"name": "NVIDIA Test GPU"}],
        },
    )

    payload = readiness.cuda_status()

    assert payload["status"] == "available"
    assert payload["backend"] == "nvidia-smi"
    assert payload["cuda_available"] is True
    assert payload["device_name"] == "NVIDIA Test GPU"
    assert payload["torch_available"] is False


def test_model_readiness_failed_requirements_lists_gpu_and_tool_failures() -> None:
    readiness = load_model_readiness_module()
    payload = {
        "tools": {"gnina": {"status": "missing", "path": None}},
        "imports": {},
        "envs": {},
        "gpu": {"status": "available", "cuda_available": False},
    }

    failures = readiness.failed_requirements(payload, ["gpu", "gnina"])

    assert [failure["requirement"] for failure in failures] == ["gpu", "gnina"]
    assert failures[0]["expected"] == "cuda_available=true"
    assert failures[1]["actual"] == {"status": "missing", "path": None}


def test_model_readiness_failed_requirements_rejects_busy_gpu() -> None:
    readiness = load_model_readiness_module()
    payload = {
        "tools": {},
        "imports": {},
        "envs": {},
        "gpu": {"status": "available", "cuda_available": True},
        "gpu_execution": {
            "status": "warning",
            "available_for_analysis": False,
            "free_mib": 512,
        },
    }

    failures = readiness.failed_requirements(payload, ["gpu", "gpu_execution"])

    assert [failure["requirement"] for failure in failures] == ["gpu_execution"]
    assert "6144 MiB" in failures[0]["expected"]


def test_diffdock_is_a_checked_requirement(monkeypatch) -> None:
    """It gates the comprehensive DAG and nothing was checking for it.

    autodock_pick_top_pct takes diffdock_blind_no_pocket's output as a required
    input, and no installer provides the command, so readiness reported a
    complete environment for a run that could not start.
    """
    module = load_model_readiness_module()

    assert module.STATUS_REQUIREMENTS["diffdock"] == ("tools", "diffdock", "available")

    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    status = module.diffdock_status()
    assert status["status"] == "missing"
    assert "comprehensive" in status["reason"]


def test_the_diffdock_check_states_the_contract_a_wrapper_must_meet() -> None:
    """The README tells a reader to write a wrapper; this is what it must accept."""
    module = load_model_readiness_module()
    status = module.diffdock_status()

    assert status["required_flags"] == [
        "--protein_path",
        "--ligand",
        "--out_dir",
        "--samples_per_complex",
        "--inference_steps",
    ]
    assert "rank1_confidence" in status["expected_output"]


def test_a_diffdock_command_on_path_is_recognised(tmp_path: Path, monkeypatch) -> None:
    module = load_model_readiness_module()
    fake = tmp_path / "diffdock"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)
    monkeypatch.setattr(
        module.shutil, "which", lambda name: str(fake) if name == "diffdock" else None
    )

    status = module.diffdock_status()

    assert status["status"] == "available"
    assert status["path"] == str(fake)


def test_the_readme_documents_every_unprovisioned_requirement() -> None:
    """The three the installers do not provide must each have real instructions.

    These live in the operator reference now: a wet-lab reader never installs
    RTMScore by hand, and 654 lines of it sat between them and the rest of the
    README. The README still has to point at the file that carries them.
    """
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/OPERATOR_CLI.md" in readme
    operator = (ROOT / "docs" / "OPERATOR_CLI.md").read_text(encoding="utf-8")
    section = operator.split("Comprehensive 모드가 추가로 요구하는 것", 1)
    assert len(section) == 2, "operator section is missing"
    body = section[1]
    for marker in ("RTMScore", "PSICHIC", "DiffDock"):
        assert marker in body, marker
    for repo in (
        "github.com/sc8668/RTMScore",
        "github.com/huankoh/PSICHIC",
        "github.com/gcorso/DiffDock",
    ):
        assert repo in body, repo
    assert "model_readiness.py" in body
    # The adapter is versioned here rather than pasted into the README, so the
    # instructions and the script it installs cannot drift apart.
    adapter = ROOT / "scripts" / "diffdock_adapter.sh"
    assert adapter.exists()
    assert "scripts/diffdock_adapter.sh" in body
    text = adapter.read_text()
    for flag in ("--protein_path", "--ligand", "--out_dir"):
        assert flag in text, flag
    assert "--ligand_description" in text, "upstream flag translation"
    assert "rank1_confidence" in text, "output relocation"
    assert "readlink -f" in text, "paths must survive the directory change"


def test_tools_in_their_snakemake_environments_count_as_available(
    tmp_path: Path, monkeypatch
) -> None:
    """Each rule activates its own conda env, so these are never on PATH.

    Reporting them missing made the browser refuse to start analyses whose
    tools were installed, because analysis readiness gates on that list.
    """
    module = load_model_readiness_module()
    envs = tmp_path / "envs" / "cosmax-qm" / "bin"
    envs.mkdir(parents=True)
    tool = envs / "xtb"
    tool.write_text("#!/bin/sh\nexit 0\n")
    tool.chmod(0o755)

    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    monkeypatch.setenv("MAMBA_ROOT_PREFIX", str(tmp_path))
    monkeypatch.delenv("SKINSCOUT_DISABLE_LOCAL_TOOL_PATHS", raising=False)

    status = module.command_status("xtb")
    assert status["status"] == "available"
    assert status["source"] == "conda_env"
    assert status["path"] == str(tool)


def test_active_environment_only_does_not_discover_inactive_conda_tools(
    tmp_path: Path, monkeypatch,
) -> None:
    module = load_model_readiness_module()
    env_bin = tmp_path / "envs/cosmax-qm/bin"
    env_bin.mkdir(parents=True)
    tool = env_bin / "xtb"
    tool.write_text("#!/bin/sh\nexit 0\n")
    tool.chmod(0o755)
    monkeypatch.setattr(module.shutil, "which", lambda _name: None)
    monkeypatch.setenv("MAMBA_ROOT_PREFIX", str(tmp_path))

    assert module.command_status("xtb")["status"] == "available"
    assert module.command_status(
        "xtb", discover_inactive_conda_envs=False
    )["status"] == "missing"


def test_active_environment_only_cli_fails_when_tool_exists_only_in_inactive_env(
    tmp_path: Path,
) -> None:
    env_bin = tmp_path / "envs/cosmax-qm/bin"
    env_bin.mkdir(parents=True)
    tool = env_bin / "xtb"
    tool.write_text("#!/bin/sh\nexit 0\n")
    tool.chmod(0o755)
    empty_path = tmp_path / "empty-path"
    empty_path.mkdir()
    env = os.environ.copy()
    env["PATH"] = str(empty_path)
    env["MAMBA_ROOT_PREFIX"] = str(tmp_path)

    discovered = subprocess.run(
        [sys.executable, str(READINESS), "--require", "xtb"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    active_only = subprocess.run(
        [
            sys.executable,
            str(READINESS),
            "--active-environment-only",
            "--require",
            "xtb",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert discovered.returncode == 0
    assert json.loads(discovered.stdout)["tools"]["xtb"]["source"] == "conda_env"
    assert active_only.returncode == 2
    assert json.loads(active_only.stdout)["tools"]["xtb"]["status"] == "missing"


def test_a_genuinely_absent_tool_is_still_missing(tmp_path: Path, monkeypatch) -> None:
    module = load_model_readiness_module()
    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    monkeypatch.setenv("MAMBA_ROOT_PREFIX", str(tmp_path))
    assert module.command_status("definitely-not-installed")["status"] == "missing"


def test_autogrid_readiness_matches_what_the_pipeline_resolves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Readiness must accept exactly the candidate set the maps step accepts.

    stage3_autogrid_maps._resolve_parameter_file takes the first non-empty file
    from an explicit path, AUTODOCK_PARAMETER_FILE, then five local candidates.
    Asserting on the source text of autogrid_status pinned one spelling of the
    rule instead of the rule, so this checks the behaviour against that list.
    """
    readiness = load_model_readiness_module()
    monkeypatch.setattr(readiness, "ROOT", tmp_path)
    monkeypatch.setattr(
        readiness,
        "command_status",
        lambda _name: {"status": "available", "path": "/fake/autogrid4"},
    )
    monkeypatch.delenv("AUTODOCK_PARAMETER_FILE", raising=False)

    for relative in (
        "tools/AutoDock-GPU/AD4_parameters.dat",
        "tools/autogrid/share/AD4.1_bound.dat",
        "tools/AutoGrid/AD4.1_bound.dat",
        "tools/AutoGrid/parameter_library/AD4.1_bound.dat",
    ):
        candidate = tmp_path / relative
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_text("parameters\n", encoding="utf-8")

        status = readiness.autogrid_status()
        assert status["status"] == "available", (
            f"the pipeline accepts {relative} but readiness rejected it"
        )
        assert status["resolved_parameter_file"] is not None

        candidate.unlink()

    assert readiness.autogrid_status()["status"] == "missing"


def test_every_registered_requirement_can_actually_be_selected() -> None:
    """`--require diffdock` was rejected by argparse before it could run.

    DiffDock is a hard input of the comprehensive DAG, so a machine without it
    reported ready and then died mid-run. The choices are derived from the
    requirement tables now, which makes that drift impossible.
    """
    module = load_model_readiness_module()
    result = subprocess.run(
        [sys.executable, str(READINESS), "--help"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    selectable = result.stdout
    for requirement in sorted(module.STATUS_REQUIREMENTS):
        assert requirement in selectable, f"--require {requirement} is unselectable"
    for requirement in sorted(module.SPECIAL_REQUIREMENTS):
        assert requirement in selectable, f"--require {requirement} is unselectable"


def test_diffdock_is_required_for_the_comprehensive_dag() -> None:
    """autodock_pick_top_pct takes diffdock_blind_no_pocket's scores as input."""
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT))
    import run_skinscout
    from workbench import server

    assert "diffdock" in run_skinscout._required_models("target-id", "comprehensive", False)
    assert "diffdock" in run_skinscout._required_models("target-id", "both", False)
    assert "diffdock" not in run_skinscout._required_models("target-id", "fast", False)
    assert "diffdock" in server.TARGET_COMPREHENSIVE_REQUIREMENTS
    assert "diffdock" not in server.TARGET_FAST_REQUIREMENTS
