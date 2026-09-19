"""F48: the core-contracts CI job really lists the audit regression modules.

The delivery job intentionally ran only three modules. This test keeps the new
dependency-light job honest: every path it names exists, the set matches the
contract modules the audit asked to run, and it does not quietly grow into a
scientific/Workbench suite that needs rdkit, biopython, torch, or a GPU.
"""

from __future__ import annotations

import shlex
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "delivery-contracts.yml"

CORE_CONTRACTS = {
    "scripts/tests/test_public_snapshot.py",
    "scripts/tests/test_readme_delivery.py",
    "scripts/tests/test_explore_target.py",
    "scripts/tests/test_qualification_manifest.py",
    "scripts/tests/test_qualification_outputs.py",
    "scripts/tests/test_coordinator_sealing.py",
    "scripts/tests/test_analog_bundle.py",
    "scripts/tests/test_stage3_band_rerank.py",
    "scripts/tests/test_build_chembl_activity_evidence.py",
    "scripts/tests/test_build_evidence_splits.py",
    "scripts/tests/test_pocket_cold_identity_regressions.py",
    "scripts/tests/test_audit_pocket_leakage.py",
    "scripts/tests/test_pocket_audit_schema.py",
    "scripts/tests/test_audit_rcsb_contact_pocket_leakage.py",
    "scripts/tests/test_build_screenable_target_cluster_map.py",
}
DELIVERY_CONTRACTS = {
    "scripts/tests/test_public_snapshot.py",
    "scripts/tests/test_readme_delivery.py",
    "scripts/tests/test_explore_target.py",
}
HEAVY_DEPENDENCIES = ("rdkit", "biopython", "torch", "scipy", "playwright")


def _workflow() -> dict:
    assert WORKFLOW.is_file(), WORKFLOW
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _run_tokens(job: dict) -> list[str]:
    tokens: list[str] = []
    for step in job["steps"]:
        command = step.get("run")
        if isinstance(command, str):
            tokens.extend(shlex.split(command))
    return tokens


def _modules(tokens: list[str]) -> set[str]:
    return {token for token in tokens if token.startswith("scripts/tests/")}


def test_the_core_job_runs_the_audit_regression_modules() -> None:
    jobs = _workflow()["jobs"]

    assert "core-contracts" in jobs
    modules = _modules(_run_tokens(jobs["core-contracts"]))

    assert modules == CORE_CONTRACTS
    for module in sorted(modules):
        assert (ROOT / module).is_file(), f"workflow lists a missing module: {module}"


def test_the_core_job_stays_dependency_light() -> None:
    job = _workflow()["jobs"]["core-contracts"]
    installs = " ".join(
        step.get("run", "") for step in job["steps"] if isinstance(step.get("run"), str)
    )
    assert "pip install" in installs
    for dependency in HEAVY_DEPENDENCIES:
        assert dependency not in installs, f"core job must not install {dependency}"

    # The Workbench server imports rdkit and biopython, so its suites cannot run
    # in this job and the workflow must not pretend they do.
    modules = _modules(_run_tokens(job))
    assert "scripts/tests/test_workbench_server.py" not in modules
    assert "scripts/tests/test_bundle_seal_reuse.py" not in modules


def test_the_delivery_job_is_left_alone() -> None:
    jobs = _workflow()["jobs"]

    assert _modules(_run_tokens(jobs["contracts"])) == DELIVERY_CONTRACTS


def test_the_workflow_still_runs_on_push_and_pull_request() -> None:
    # PyYAML parses the unquoted `on:` key as boolean True.
    triggers = _workflow().get(True) or _workflow().get("on")
    assert "push" in triggers
    assert "pull_request" in triggers
