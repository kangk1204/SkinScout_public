#!/usr/bin/env python3
"""Generate and browser-seal a deterministic report from structural smoke output."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "skinscout.report-qualification.v1"
ADMET_PREDICTIONS = {
    "molecular_weight": 46.069,
    "logP": -0.0014,
    "QED": 0.4068,
    "tpsa": 20.23,
    "AMES": 0.04,
    "ClinTox": 0.01,
    "DILI": 0.09,
    "Skin_Reaction": 0.21,
    "hERG": 0.01,
    "Caco2_Wang": -4.0,
    "LD50_Zhu": 1.1,
    "Solubility_AqSolDB": 1.2,
}
STRUCTURAL_ALERTS = {
    "smiles": "CCO",
    "matches": {"PAINS_A": [], "PAINS_B": [], "PAINS_C": [], "BRENK": [], "NIH": []},
    "any_pains": False,
    "any_brenk": False,
    "any_nih": False,
}


class SmokeError(RuntimeError):
    """A report qualification step failed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SmokeError(f"{label} is unreadable: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SmokeError(f"{label} must contain a JSON object: {path}")
    return payload


def _write_safety_inputs(input_dir: Path) -> dict[str, Path]:
    input_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        name: input_dir / filename
        for name, filename in {
            "compound": "compound.json",
            "admet": "admet.json",
            "skin_sens_decision": "skin_sens_decision.txt",
            "admet_ai": "admet_ai.json",
            "structural_alerts": "structural_alerts.json",
            "husspred": "husspred.json",
            "stoptox": "stoptox.json",
            "pred_skin": "pred_skin.json",
            "cosmetic_decision": "cosmetic_drug_decision.txt",
            "drug_warnings": "drug_warnings.json",
        }.items()
    }
    _write_json(
        paths["compound"],
        {
            "input_type": "smiles",
            "input_smiles": "CCO",
            "input_canonical_smiles": "CCO",
            "canonical_smiles": "CCO",
            "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        },
    )
    calls = {"husspred": "negative", "stoptox": "negative", "pred_skin": "negative"}
    _write_json(
        paths["admet"],
        {
            "skin_sens": {
                **calls,
                "decision": "PASS",
                "degraded": False,
                "missing_models": [],
            },
            "structural_alerts": STRUCTURAL_ALERTS,
            "admet_ai_predictions": ADMET_PREDICTIONS,
        },
    )
    paths["skin_sens_decision"].write_text("PASS\n", encoding="utf-8")
    _write_json(
        paths["admet_ai"],
        {"smiles": "CCO", "status": "ok", "predictions": ADMET_PREDICTIONS},
    )
    _write_json(paths["structural_alerts"], STRUCTURAL_ALERTS)
    for name in ("husspred", "stoptox"):
        _write_json(
            paths[name],
            {
                "smiles": "CCO",
                "status": "ok",
                "skin_sens_probability": 0.2,
                "skin_sens_call": "negative",
            },
        )
    _write_json(
        paths["pred_skin"],
        {
            "smiles": "CCO",
            "status": "ok",
            "consensus_call": "negative",
            "pred_skin": {"probability": 0.2, "call": "negative"},
        },
    )
    paths["cosmetic_decision"].write_text(
        "PROCEED\n# policy: moderate\n",
        encoding="utf-8",
    )
    _write_json(
        paths["drug_warnings"],
        {
            "max_tanimoto_to_approved_drug": 0.1,
            "warnings": [],
            "n_warnings": 0,
            "reference_status": "ok",
        },
    )
    return paths


def run_smoke(structural_dir: Path, browser: Path) -> dict[str, Any]:
    structural_status = _load_json(
        structural_dir / "qualification_status.json",
        "structural qualification status",
    )
    if structural_status.get("status") != "passed":
        raise SmokeError("structural qualification must pass before report qualification")
    run_dir = structural_dir / "run"
    fast_dir = run_dir / "03_targets/mode_fast"
    paths = _write_safety_inputs(run_dir / "qualification_report_inputs")
    ligand_sdf = run_dir / "01_input/qualification_ligand.sdf"
    ligand_pdbqt = run_dir / "03_targets/mode_comprehensive/ligand.pdbqt"
    pose_dir = fast_dir / "docked_poses"
    stage3_top = fast_dir / "daina_structural_targets.csv"
    autodock = fast_dir / "autodock_top5k.tsv"
    report_dir = run_dir / "09_report"
    report_html = report_dir / "index.html"
    fast_manifest = report_dir / "fast_report_manifest.json"
    physics_manifest = report_dir / "physics_report_manifest.json"
    command = [
        sys.executable,
        str(ROOT / "scripts/stage9_report.py"),
        "--run-id",
        "installer-qualification",
        "--mode",
        "fast",
        "--compound-meta",
        str(paths["compound"]),
        "--admet-report",
        str(paths["admet"]),
        "--skin-sens-decision",
        str(paths["skin_sens_decision"]),
        "--admet-ai-json",
        str(paths["admet_ai"]),
        "--structural-alerts-json",
        str(paths["structural_alerts"]),
        "--husspred-json",
        str(paths["husspred"]),
        "--stoptox-json",
        str(paths["stoptox"]),
        "--pred-skin-json",
        str(paths["pred_skin"]),
        "--stage3-top",
        str(stage3_top),
        "--cosmetic-drug-decision",
        str(paths["cosmetic_decision"]),
        "--drug-warnings-json",
        str(paths["drug_warnings"]),
        "--autodock-full",
        str(autodock),
        "--ligand-sdf",
        str(ligand_sdf),
        "--ligand-pdbqt",
        str(ligand_pdbqt),
        "--pose-dir",
        str(pose_dir),
        "--receptor-dir",
        str(ROOT / "data/human_clean"),
        "--fast-report-root",
        str(run_dir / "reports/fast"),
        "--physics-report-root",
        str(run_dir / "reports/physics"),
        "--fast-manifest-output",
        str(fast_manifest),
        "--physics-manifest-output",
        str(physics_manifest),
        "--browser-path",
        str(browser),
        "--run-dir",
        str(run_dir),
        "--out-html",
        str(report_html),
    ]
    log_path = structural_dir / "report_commands.log"
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    log_path.write_text(
        "$ " + shlex.join(command) + "\n" + result.stdout + result.stderr + f"\n[exit={result.returncode}]\n",
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise SmokeError(f"report-fast qualification failed; see {log_path}")
    manifest = _load_json(fast_manifest, "fast-report manifest")
    browser_verification = manifest.get("browser_verification")
    if (
        manifest.get("sealed") is not True
        or not isinstance(browser_verification, dict)
        or browser_verification.get("status") != "passed"
    ):
        raise SmokeError("report-fast package was generated but not browser sealed")
    if not report_html.is_file() or report_html.stat().st_size == 0:
        raise SmokeError("report-fast compatibility HTML is missing or empty")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "browser": str(browser),
        "fast_report_artifact_id": manifest.get("artifact_id"),
        "browser_verification": browser_verification,
        "outputs": {
            "report_html": {
                "path": str(report_html.relative_to(structural_dir)),
                "bytes": report_html.stat().st_size,
                "sha256": _sha256(report_html),
            },
            "fast_manifest": {
                "path": str(fast_manifest.relative_to(structural_dir)),
                "bytes": fast_manifest.stat().st_size,
                "sha256": _sha256(fast_manifest),
            },
        },
    }
    _write_json(structural_dir / "report_qualification_status.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structural-dir", required=True, type=Path)
    parser.add_argument("--browser", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    structural_dir = args.structural_dir.expanduser().resolve()
    browser = args.browser.expanduser().resolve()
    if not browser.is_file() or not os.access(browser, os.X_OK):
        raise SmokeError(f"Chromium executable is unavailable: {browser}")
    status_path = structural_dir / "report_qualification_status.json"
    _write_json(status_path, {"schema_version": SCHEMA_VERSION, "status": "running"})
    try:
        payload = run_smoke(structural_dir, browser)
    except Exception as exc:
        _write_json(
            status_path,
            {"schema_version": SCHEMA_VERSION, "status": "failed", "detail": str(exc)},
        )
        raise
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
