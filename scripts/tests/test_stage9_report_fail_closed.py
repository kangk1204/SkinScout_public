"""Regression tests for Stage 9 report required artifact gates."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
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


def test_preformatted_json_escapes_stored_html(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "stage9_report_html_escape_test",
        ROOT / "scripts" / "stage9_report.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    rendered = module._preformatted_json({"value": "</pre><script>alert(1)</script>"})

    assert "<script>" not in rendered
    assert "&lt;/pre&gt;&lt;script&gt;alert(1)&lt;/script&gt;" in rendered
SKIN_SENS_CALLS = {
    "husspred": "negative",
    "stoptox": "negative",
    "pred_skin": "negative",
}


def test_fast_daina_report_preserves_unknown_efficacy_without_weakening_comprehensive(monkeypatch) -> None:
    import pandas as pd
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))

    spec = importlib.util.spec_from_file_location("stage9_unknown_efficacy", ROOT / "scripts/stage9_report.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    frame = pd.DataFrame([{
        "target_id": "P12345", "final_score": 0.8, "skin_score": 0.5,
        "source_count": 3, "sources": "a;b;c", "efficacy_top1": "",
    }])
    kwargs = {"label": "test", "path": Path("ranking.csv")}
    module._validate_stage3_ranking_contract(frame, mode="fast", allow_missing_efficacy=True, **kwargs)
    for mode in ("fast", "comprehensive"):
        with pytest.raises(SystemExit, match="lacks KG skin-efficacy"):
            module._validate_stage3_ranking_contract(
                frame, mode=mode, allow_missing_efficacy=(mode == "comprehensive"), **kwargs
            )


def write_test_sdf(
    path: Path,
    *,
    y_offset: float = 0.0,
    target_id: str | None = None,
    docking_energy: float | None = None,
) -> None:
    properties = ""
    if target_id is not None:
        properties += f">  <target_id>\n{target_id}\n\n"
    if docking_energy is not None:
        properties += (
            f">  <docking_energy_kcal_mol>\n{docking_energy:.3f}\n\n"
        )
    path.write_text(
        "ethanol\n"
        "  SkinScout\n\n"
        "  3  2  0  0  0  0            999 V2000\n"
        f"    0.0000    {y_offset:0.4f}    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        f"    1.5000    {y_offset:0.4f}    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        f"    2.8000    {y_offset:0.4f}    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "  1  2  1  0  0  0  0\n"
        "  2  3  1  0  0  0  0\n"
        "M  END\n"
        + properties
        + "$$$$\n"
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_pose_manifest(paths: dict[str, Path]) -> None:
    rows = []
    score_lines = paths["autodock"].read_text().splitlines()
    header = score_lines[0].split("\t")
    for line in score_lines[1:]:
        values = dict(zip(header, line.split("\t"), strict=True))
        target_id = values["target_id"]
        pose = paths["pose_dir"] / f"{target_id}.sdf"
        rows.append({
            "target_id": target_id,
            "docking_energy_kcal_mol": float(values["vina_score"]),
            "pose_file": pose.name,
            "pose_sha256": sha256_file(pose),
            "pose_bytes": pose.stat().st_size,
        })
    (paths["pose_dir"] / "pose_manifest.json").write_text(json.dumps({
        "schema_version": "skinscout.docking_pose_manifest.v1",
        "engine": "autodock_gpu",
        "score_file": paths["autodock"].name,
        "score_sha256": sha256_file(paths["autodock"]),
        "score_bytes": paths["autodock"].stat().st_size,
        "target_count": len(rows),
        "targets": rows,
    }, indent=2, sort_keys=True) + "\n")


def run_script(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def probability_for_call(call: str) -> float:
    return 0.8 if call == "positive" else 0.2


def write_admet_bundle(
    paths: dict[str, Path],
    *,
    predictions: dict[str, float] | None = None,
    structural_alerts: dict[str, object] | None = None,
    skin_sens_calls: dict[str, str] | None = None,
    decision: str = "PASS",
) -> None:
    admet_predictions = predictions or dict(ADMET_PREDICTIONS)
    alerts = structural_alerts or dict(STRUCTURAL_ALERTS)
    calls = skin_sens_calls or dict(SKIN_SENS_CALLS)
    paths["admet"].write_text(json.dumps({
        "skin_sens": {
            "husspred": calls["husspred"],
            "stoptox": calls["stoptox"],
            "pred_skin": calls["pred_skin"],
            "decision": decision,
            "degraded": False,
            "missing_models": [],
        },
        "structural_alerts": alerts,
        "admet_ai_predictions": admet_predictions,
    }) + "\n")
    paths["skin_sens_decision"].write_text(decision + "\n")
    paths["admet_ai"].write_text(json.dumps({
        "smiles": "CCO",
        "status": "ok",
        "predictions": admet_predictions,
    }) + "\n")
    paths["structural_alerts"].write_text(json.dumps(alerts) + "\n")
    paths["husspred"].write_text(json.dumps({
        "smiles": "CCO",
        "status": "ok",
        "skin_sens_probability": probability_for_call(calls["husspred"]),
        "skin_sens_call": calls["husspred"],
    }) + "\n")
    paths["stoptox"].write_text(json.dumps({
        "smiles": "CCO",
        "status": "ok",
        "skin_sens_probability": probability_for_call(calls["stoptox"]),
        "skin_sens_call": calls["stoptox"],
    }) + "\n")
    paths["pred_skin"].write_text(json.dumps({
        "smiles": "CCO",
        "status": "ok",
        "consensus_call": calls["pred_skin"],
        "pred_skin": {
            "probability": probability_for_call(calls["pred_skin"]),
            "call": calls["pred_skin"],
        },
    }) + "\n")


def write_required_inputs(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "compound": tmp_path / "compound.json",
        "admet": tmp_path / "admet.json",
        "skin_sens_decision": tmp_path / "skin_sens_decision.txt",
        "admet_ai": tmp_path / "admet_ai.json",
        "structural_alerts": tmp_path / "structural_alerts.json",
        "husspred": tmp_path / "husspred.json",
        "stoptox": tmp_path / "stoptox.json",
        "pred_skin": tmp_path / "pred_skin.json",
        "stage3": tmp_path / "stage3.csv",
        "cosmetic_decision": tmp_path / "cosmetic_drug_decision.txt",
        "drug_warnings": tmp_path / "drug_warnings.json",
        "autodock": tmp_path / "autodock.tsv",
        "ligand_sdf": tmp_path / "ligand.sdf",
        "ligand_pdbqt": tmp_path / "ligand.pdbqt",
        "pose_dir": tmp_path / "poses",
        "receptor_dir": tmp_path / "receptors",
        "molstar_bundle": tmp_path / "molstar.js",
        "molstar_stylesheet": tmp_path / "molstar.css",
        "boltz": tmp_path / "boltz.tsv",
        "ensemble": tmp_path / "ensemble.tsv",
        "mmgbsa": tmp_path / "mmgbsa.tsv",
        "dft": tmp_path / "dft.tsv",
        "html": tmp_path / "report" / "index.html",
    }
    paths["compound"].write_text(
        '{"input_type": "smiles", "input_smiles": "C(C)O", '
        '"input_canonical_smiles": "CCO", '
        '"inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N", '
        '"canonical_smiles": "CCO"}\n'
    )
    write_admet_bundle(paths)
    paths["stage3"].write_text(
        "target_id,final_score,skin_score,skin_tier,source_count,sources,efficacy_top1\n"
        "P1,1.0,0.8,high,3,autodock;gnina;rtmscore,hydration (12 papers)\n"
    )
    comprehensive_dir = tmp_path / "03_targets" / "mode_comprehensive"
    comprehensive_dir.mkdir(parents=True)
    fast_dir = tmp_path / "03_targets" / "mode_fast"
    fast_dir.mkdir(parents=True)
    (fast_dir / "psichic_proteome.tsv").write_text("target_id\tscore\nP1\t0.9\n")
    (fast_dir / "daina_zoete_proteome.tsv").write_text("target_id\tscore\nP1\t0.8\n")
    (fast_dir / "dti_rrf_top25pct.csv").write_text("target_id,score\nP1,0.7\n")
    (fast_dir / "autodock_top5k.tsv").write_text(
        "target_id\tvina_score\tneg_vina_score\nP1\t-7.0\t7.0\n"
    )
    (fast_dir / "top50.csv").write_text(
        "target_id,final_score,skin_score,source_count,sources,efficacy_top1\n"
        "P1,1.0,0.8,2,autodock;psichic,hydration (12 papers)\n"
    )
    (comprehensive_dir / "autodock_all_targets.tsv").write_text(
        "target_id\tvina_score\tneg_vina_score\nP1\t-7.0\t7.0\n"
    )
    (comprehensive_dir / "top_pct_pre_rescore.csv").write_text(
        "target_id,score,source\nP1,7.0,autodock\n"
    )
    (comprehensive_dir / "gnina_rescores.tsv").write_text(
        "target_id\tcnn_affinity\tscore\nP1\t8.0\t8.0\n"
    )
    (comprehensive_dir / "rtmscore_rescores.tsv").write_text(
        "target_id\trtm_score\tscore\nP1\t0.8\t0.8\n"
    )
    (comprehensive_dir / "boltz2_affinity_top.tsv").write_text(
        "target_id\tboltz2_neg_log_uM\tscore\nP1\t6.5\t6.5\n"
    )
    (comprehensive_dir / "top50_4way_consensus.csv").write_text(
        "target_id,rrf_score,source_count,sources\n"
        "P1,0.9,4,autodock;boltz;gnina;rtm\n"
    )
    paths["cosmetic_decision"].write_text("PROCEED\n# policy: moderate\n")
    paths["drug_warnings"].write_text(
        '{"max_tanimoto_to_approved_drug": 0.1, "warnings": [], '
        '"n_warnings": 0, "reference_status": "ok"}\n'
    )
    paths["autodock"].write_text(
        "target_id\tvina_score\tneg_vina_score\nP1\t-7.0\t7.0\n"
    )
    write_test_sdf(paths["ligand_sdf"])
    paths["ligand_pdbqt"].write_text("ROOT\nENDROOT\nTORSDOF 0\n")
    paths["pose_dir"].mkdir()
    write_test_sdf(
        paths["pose_dir"] / "P1.sdf",
        y_offset=1.0,
        target_id="P1",
        docking_energy=-7.0,
    )
    paths["receptor_dir"].mkdir()
    (paths["receptor_dir"] / "P1_clean.pdb").write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n"
        "END\n"
    )
    paths["molstar_bundle"].write_text(
        "window.molstar={Viewer:{create:async()=>({plugin:{clear:async()=>{}},"
        "loadStructureFromUrl:async()=>{}})}};\n"
    )
    paths["molstar_stylesheet"].write_text("#molstar-viewer{position:relative}\n")
    paths["boltz"].write_text("target_id\tkept\nP1\tyes\n")
    paths["ensemble"].write_text("target_id\tconsensus_score\nP1\t2.5\n")
    paths["mmgbsa"].write_text(
        "target_id\tmmgbsa_dg_kcal_mol\tstatus\nP1\t-12.3\tok\n"
    )
    paths["dft"].write_text(
        "target_id\tdft_energy_hartree\tstatus\nP1\t-123.4\tok\n"
    )
    write_pose_manifest(paths)
    return paths


def run_stage9(
    tmp_path: Path,
    paths: dict[str, Path],
    extra_args: list[str] | None = None,
    include_fast_asset_args: bool = True,
) -> subprocess.CompletedProcess[str]:
    fast_asset_args = [
        "--ligand-sdf", str(paths["ligand_sdf"]),
        "--ligand-pdbqt", str(paths["ligand_pdbqt"]),
        "--pose-dir", str(paths["pose_dir"]),
        "--receptor-dir", str(paths["receptor_dir"]),
        "--molstar-bundle", str(paths["molstar_bundle"]),
        "--molstar-stylesheet", str(paths["molstar_stylesheet"]),
    ] if include_fast_asset_args else []
    return run_script([
        "scripts/stage9_report.py",
        "--run-id", "test-run",
        "--compound-meta", str(paths["compound"]),
        "--admet-report", str(paths["admet"]),
        "--skin-sens-decision", str(paths["skin_sens_decision"]),
        "--admet-ai-json", str(paths["admet_ai"]),
        "--structural-alerts-json", str(paths["structural_alerts"]),
        "--husspred-json", str(paths["husspred"]),
        "--stoptox-json", str(paths["stoptox"]),
        "--pred-skin-json", str(paths["pred_skin"]),
        "--stage3-top", str(paths["stage3"]),
        "--cosmetic-drug-decision", str(paths["cosmetic_decision"]),
        "--drug-warnings-json", str(paths["drug_warnings"]),
        "--autodock-full", str(paths["autodock"]),
        *fast_asset_args,
        "--boltz-report", str(paths["boltz"]),
        "--ensemble", str(paths["ensemble"]),
        "--mmgbsa", str(paths["mmgbsa"]),
        "--dft", str(paths["dft"]),
        "--run-dir", str(tmp_path),
        "--out-html", str(paths["html"]),
        "--skip-browser-verification",
        *(extra_args or []),
    ])


def run_stage9_fast(
    tmp_path: Path,
    paths: dict[str, Path],
    extra_args: list[str] | None = None,
    *,
    use_vendored_molstar: bool = False,
    skip_browser_verification: bool = True,
) -> subprocess.CompletedProcess[str]:
    viewer_args = [] if use_vendored_molstar else [
        "--molstar-bundle", str(paths["molstar_bundle"]),
        "--molstar-stylesheet", str(paths["molstar_stylesheet"]),
    ]
    browser_args = (
        ["--skip-browser-verification"] if skip_browser_verification else []
    )
    return run_script([
        "scripts/stage9_report.py",
        "--run-id", "test-run",
        "--mode", "fast",
        "--compound-meta", str(paths["compound"]),
        "--admet-report", str(paths["admet"]),
        "--skin-sens-decision", str(paths["skin_sens_decision"]),
        "--admet-ai-json", str(paths["admet_ai"]),
        "--structural-alerts-json", str(paths["structural_alerts"]),
        "--husspred-json", str(paths["husspred"]),
        "--stoptox-json", str(paths["stoptox"]),
        "--pred-skin-json", str(paths["pred_skin"]),
        "--stage3-top", str(paths["stage3"]),
        "--cosmetic-drug-decision", str(paths["cosmetic_decision"]),
        "--drug-warnings-json", str(paths["drug_warnings"]),
        "--autodock-full", str(paths["autodock"]),
        "--ligand-sdf", str(paths["ligand_sdf"]),
        "--ligand-pdbqt", str(paths["ligand_pdbqt"]),
        "--pose-dir", str(paths["pose_dir"]),
        "--receptor-dir", str(paths["receptor_dir"]),
        *viewer_args,
        "--run-dir", str(tmp_path),
        "--out-html", str(paths["html"]),
        *browser_args,
        *(extra_args or []),
    ])


def add_second_fast_target(paths: dict[str, Path], tmp_path: Path) -> None:
    paths["stage3"].write_text(
        "target_id,final_score,skin_score,skin_tier,source_count,sources,efficacy_top1\n"
        "P1,1.0,0.8,high,3,autodock;gnina;rtmscore,hydration (12 papers)\n"
        "P2,0.9,0.7,high,3,autodock;gnina;rtmscore,barrier (8 papers)\n"
    )
    paths["autodock"].write_text(
        "target_id\tvina_score\tneg_vina_score\n"
        "P1\t-7.0\t7.0\n"
        "P2\t-6.5\t6.5\n"
    )
    fast_dir = tmp_path / "03_targets" / "mode_fast"
    (fast_dir / "autodock_top5k.tsv").write_text(paths["autodock"].read_text())
    (fast_dir / "top50.csv").write_text(
        "target_id,final_score,skin_score,source_count,sources,efficacy_top1\n"
        "P1,1.0,0.8,2,autodock;psichic,hydration (12 papers)\n"
        "P2,0.9,0.7,2,autodock;psichic,barrier (8 papers)\n"
    )
    write_test_sdf(
        paths["pose_dir"] / "P2.sdf",
        y_offset=2.0,
        target_id="P2",
        docking_energy=-6.5,
    )
    (paths["receptor_dir"] / "P2_clean.pdb").write_text(
        "ATOM      1  CA  GLY A   2       1.000   1.000   1.000  1.00 20.00           C\n"
        "END\n"
    )
    write_pose_manifest(paths)


def test_stage9_fast_report_succeeds_without_physics_and_writes_immutable_package(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)

    res = run_stage9_fast(tmp_path, paths)

    assert res.returncode == 0, res.stderr
    assert paths["html"].exists()
    stub = paths["html"].read_text()
    assert "Immutable Fast Report" in stub
    assert "http://" not in stub
    assert "https://" not in stub

    fast_dirs = sorted((tmp_path / "reports" / "fast").iterdir())
    assert len(fast_dirs) == 1
    package = fast_dirs[0]
    manifest = json.loads((package / "manifest.json").read_text())
    checksums = json.loads((package / "checksums.json").read_text())
    assert manifest["schema_version"] == "skinscout.report_fast.v1"
    assert manifest["artifact_id"] == package.name
    assert manifest["sealed"] is False
    assert manifest["browser_verification"]["status"] == "skipped_test_only"
    assert manifest["viewer"]["name"] == "Mol*"
    assert manifest["viewer"]["runtime_network"] is False
    assert manifest["parent_linkage"]["fast_parent_hash"] == package.name
    assert manifest["parent_linkage"]["physics_parent_hash"] is None
    assert manifest["parent_linkage"]["physics_parent_status"] == "not_requested"
    checksum_paths = {entry["path"] for entry in checksums["files"]}
    assert {
        "assets/molstar.js",
        "assets/molstar.css",
        "assets/report.css",
        "assets/report.js",
        "index.html",
        "interactions/P1.json",
        "molecules/ligand.sdf",
        "molecules/P1/pose.sdf",
        "molecules/P1/receptor.pdb",
        "ranked_targets.csv",
        "targets.json",
    } <= checksum_paths
    report_html = (package / "index.html").read_text()
    assert "assets/molstar.js" in report_html
    assert "assets/molstar.css" in report_html
    assert "assets/report.js" in report_html
    assert "unsafe-inline" not in report_html
    assert "http://" not in report_html
    assert "https://" not in report_html
    assert all("path" not in record for record in manifest["source_hashes"].values())
    fast_pointer = json.loads((paths["html"].parent / "fast_report_manifest.json").read_text())
    physics_pointer = json.loads(
        (paths["html"].parent / "physics_report_manifest.json").read_text()
    )
    assert fast_pointer["status"] == "generated_unverified"
    assert fast_pointer["artifact_id"] == package.name
    assert physics_pointer["status"] == "not_requested"


def test_stage9_fast_report_manifest_is_deterministic(tmp_path: Path) -> None:
    paths = write_required_inputs(tmp_path)

    first = run_stage9_fast(tmp_path, paths)
    assert first.returncode == 0, first.stderr
    package = next((tmp_path / "reports" / "fast").iterdir())
    manifest_text = (package / "manifest.json").read_text()
    checksums_text = (package / "checksums.json").read_text()

    second = run_stage9_fast(tmp_path, paths)
    assert second.returncode == 0, second.stderr
    package_after = next((tmp_path / "reports" / "fast").iterdir())
    assert package_after.name == package.name
    assert (package_after / "manifest.json").read_text() == manifest_text
    assert (package_after / "checksums.json").read_text() == checksums_text


def test_stage9_fast_artifact_id_is_independent_of_absolute_input_root(
    tmp_path: Path,
) -> None:
    roots = [tmp_path / "first", tmp_path / "second"]
    artifact_ids: list[str] = []
    for root in roots:
        root.mkdir()
        paths = write_required_inputs(root)
        result = run_stage9_fast(root, paths)
        assert result.returncode == 0, result.stderr
        artifact_ids.append(next((root / "reports" / "fast").iterdir()).name)

    assert artifact_ids[0] == artifact_ids[1]


def test_stage9_fast_report_rejects_tampered_existing_manifest(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    first = run_stage9_fast(tmp_path, paths)
    assert first.returncode == 0, first.stderr
    package = next((tmp_path / "reports" / "fast").iterdir())
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sealed"] = True
    manifest_path.write_text(json.dumps(manifest) + "\n")

    second = run_stage9_fast(tmp_path, paths)

    assert second.returncode != 0
    assert "Report manifest field is not identity-bound: sealed" in second.stderr


def test_stage9_publish_rejects_new_unsealed_package_when_seal_is_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "stage9_report_publish_test",
        ROOT / "scripts/stage9_report.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    (temporary / "payload.txt").write_text("payload\n")
    manifest_core = {
        "schema_version": module.FAST_REPORT_SCHEMA_VERSION,
        "sealed": False,
        "checksums_path": "checksums.json",
    }
    module._write_json_deterministic(temporary / "identity.json", {
        "schema_version": module.REPORT_IDENTITY_SCHEMA_VERSION,
        "kind": "fast",
        "manifest_core": manifest_core,
    })
    checksum_schema = "skinscout.report_fast.checksums.v1"
    checksums = {
        "schema_version": checksum_schema,
        "files": module._payload_files(temporary),
    }
    artifact_id = module._artifact_id(checksums)
    module._write_json_deterministic(temporary / "checksums.json", checksums)
    module._write_json_deterministic(temporary / "manifest.json", {
        **manifest_core,
        "artifact_id": artifact_id,
        "identity_path": "identity.json",
        "identity_sha256": module._sha256_file(temporary / "identity.json"),
        "parent_linkage": {"fast_parent_hash": artifact_id},
    })
    destination = tmp_path / artifact_id

    with pytest.raises(SystemExit, match="not browser-sealed"):
        module._publish_immutable_package(
            temporary,
            destination,
            artifact_id=artifact_id,
            checksum_schema=checksum_schema,
            require_sealed=True,
        )

    assert not destination.exists()


def test_stage9_fast_report_requires_real_pose_sdf_not_input_pdbqt(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    (paths["pose_dir"] / "P1.sdf").unlink()
    paths["ligand_pdbqt"].write_text(
        "ROOT\nATOM      1  C   LIG     1       9.000   9.000   9.000  1.00  0.00 C\nENDROOT\n"
    )

    result = run_stage9_fast(tmp_path, paths)

    assert result.returncode != 0
    assert "Docking pose manifest file is missing or unsafe" in result.stderr
    assert "P1.sdf" in result.stderr
    assert not (tmp_path / "reports" / "fast").exists() or not any(
        (tmp_path / "reports" / "fast").iterdir()
    )


@pytest.mark.parametrize(
    ("target_id", "energy", "message"),
    [
        ("P2", -7.0, "target_id mismatch"),
        ("P1", -6.0, "docking energy mismatch"),
    ],
)
def test_stage9_fast_report_rejects_pose_metadata_not_bound_to_score(
    tmp_path: Path,
    target_id: str,
    energy: float,
    message: str,
) -> None:
    paths = write_required_inputs(tmp_path)
    write_test_sdf(
        paths["pose_dir"] / "P1.sdf",
        y_offset=1.0,
        target_id=target_id,
        docking_energy=energy,
    )
    write_pose_manifest(paths)

    result = run_stage9_fast(tmp_path, paths)

    assert result.returncode != 0
    assert message in result.stderr


def test_stage9_physics_package_is_exactly_linked_without_mutating_fast_tree(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)

    result = run_stage9(tmp_path, paths)

    assert result.returncode == 0, result.stderr
    fast_pointer = json.loads(
        (paths["html"].parent / "fast_report_manifest.json").read_text()
    )
    physics_pointer = json.loads(
        (paths["html"].parent / "physics_report_manifest.json").read_text()
    )
    fast_root = tmp_path / "reports" / "fast" / fast_pointer["artifact_id"]
    physics_root = tmp_path / "reports" / "physics" / physics_pointer["artifact_id"]
    fast_manifest = json.loads((fast_root / "manifest.json").read_text())
    physics_manifest = json.loads((physics_root / "manifest.json").read_text())
    physics_summary = json.loads((physics_root / "summary.json").read_text())

    assert physics_pointer["parent_fast_hash"] == fast_pointer["artifact_id"]
    assert physics_manifest["parent_fast_hash"] == fast_pointer["artifact_id"]
    assert physics_summary["parent_fast_hash"] == fast_pointer["artifact_id"]
    assert fast_manifest["parent_linkage"]["physics_parent_hash"] is None
    assert not (fast_root / "data").exists()
    assert (physics_root / "data" / "boltz.tsv").is_file()
    assert (physics_root / "identity.json").is_file()
    assert physics_pointer["parent_manifest_path"] == (
        f"reports/fast/{fast_pointer['artifact_id']}/manifest.json"
    )


@pytest.mark.skipif(
    importlib.util.find_spec("playwright") is None
    or shutil.which("google-chrome") is None,
    reason="Playwright and local Google Chrome are required for the browser seal gate",
)
def test_stage9_vendored_molstar_seals_two_target_desktop_mobile_report(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    add_second_fast_target(paths, tmp_path)

    result = run_stage9_fast(
        tmp_path,
        paths,
        use_vendored_molstar=True,
        skip_browser_verification=False,
    )

    assert result.returncode == 0, result.stderr
    pointer = json.loads(
        (paths["html"].parent / "fast_report_manifest.json").read_text()
    )
    package = tmp_path / "reports" / "fast" / pointer["artifact_id"]
    manifest = json.loads((package / "manifest.json").read_text())
    verification = manifest["browser_verification"]
    assert pointer["status"] == "ready"
    assert manifest["sealed"] is True
    assert manifest["target_count"] == 2
    assert verification["status"] == "passed"
    assert verification["viewports"]["desktop"]["target_switch"] == "passed"
    assert verification["viewports"]["mobile"]["target_switch"] == "passed"
    assert verification["webgl_fallback"]["webgl_unavailable"] == "passed"


def test_stage9_fast_report_fails_without_local_molstar_bundle(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    paths["molstar_bundle"].unlink()
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9_fast(tmp_path, paths)

    assert res.returncode != 0
    assert "Local Mol* browser bundle is required" in res.stderr
    assert "CDN/runtime network loading is forbidden" in res.stderr
    assert not paths["html"].exists()


def test_stage9_fast_report_fails_without_required_receptor_asset(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    (paths["receptor_dir"] / "P1_clean.pdb").unlink()
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9_fast(tmp_path, paths)

    assert res.returncode != 0
    assert "Required receptor asset for top fast target is missing" in res.stderr
    assert not paths["html"].exists()


def test_stage9_comprehensive_preserves_legacy_cli_without_fast_assets(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)

    res = run_stage9(tmp_path, paths, include_fast_asset_args=False)

    assert res.returncode == 0, res.stderr
    assert paths["html"].exists()
    assert "Panel F" in paths["html"].read_text()
    assert not (tmp_path / "reports" / "fast").exists()


def test_stage9_report_fails_when_required_table_missing(tmp_path: Path) -> None:
    paths = write_required_inputs(tmp_path)
    paths["mmgbsa"].unlink()
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths)

    assert res.returncode != 0
    assert "MM-GBSA report is required and must be non-empty" in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_fails_when_required_table_empty(tmp_path: Path) -> None:
    paths = write_required_inputs(tmp_path)
    paths["dft"].write_text("")
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths)

    assert res.returncode != 0
    assert "DFT report is required and must be non-empty" in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_fails_when_screening_count_artifact_missing(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    (
        tmp_path
        / "03_targets"
        / "mode_comprehensive"
        / "gnina_rescores.tsv"
    ).unlink()
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths)

    assert res.returncode != 0
    assert "Target screening count gnina_rescored_targets is required" in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_fails_when_required_columns_missing(tmp_path: Path) -> None:
    paths = write_required_inputs(tmp_path)
    paths["stage3"].write_text("score\n1.0\n")
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths)

    assert res.returncode != 0
    assert "Stage 3 ranked targets missing required columns" in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_fails_when_stage3_score_nonnumeric(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    paths["stage3"].write_text(
        "target_id,final_score,skin_score,source_count,sources,efficacy_top1\n"
        "P1,not-a-score,0.8,3,autodock;gnina;rtmscore,hydration (12 papers)\n"
    )
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths)

    assert res.returncode != 0
    assert "Stage 3 ranked targets column 'final_score' must be numeric" in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_fails_when_autodock_score_columns_missing(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    paths["autodock"].write_text("target_id\tscore\nP1\t7.0\n")
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths)

    assert res.returncode != 0
    assert "AutoDock-GPU full scores missing required columns" in res.stderr
    assert "neg_vina_score" in res.stderr
    assert "vina_score" in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_fails_when_autodock_scores_nonnumeric(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    paths["autodock"].write_text(
        "target_id\tvina_score\tneg_vina_score\nP1\tbad\t7.0\n"
    )
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths)

    assert res.returncode != 0
    assert "AutoDock-GPU full scores column 'vina_score' must be numeric" in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_fails_when_autodock_score_boolean(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    paths["autodock"].write_text(
        "target_id\tvina_score\tneg_vina_score\nP1\tTrue\t7.0\n"
    )
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths)

    assert res.returncode != 0
    assert "AutoDock-GPU full scores column 'vina_score' must be numeric" in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_fails_when_compound_identity_missing(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    paths["compound"].write_text('{"inchikey": " ", "canonical_smiles": "CCO"}\n')
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths)

    assert res.returncode != 0
    assert "Compound metadata missing non-empty string field 'inchikey'" in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_fails_when_invalid_compound_smiles(tmp_path: Path) -> None:
    paths = write_required_inputs(tmp_path)
    paths["compound"].write_text(
        '{"inchikey": "TEST", "canonical_smiles": "not a smiles"}\n'
    )
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths)

    assert res.returncode != 0
    assert (
        "Compound metadata field 'canonical_smiles' contains invalid SMILES"
        in res.stderr
    )
    assert not paths["html"].exists()


def test_stage9_report_fails_when_compound_inchikey_mismatches_smiles(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    paths["compound"].write_text(
        '{"inchikey": "WRONG-INCHIKEY", "canonical_smiles": "CCO"}\n'
    )
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths)

    assert res.returncode != 0
    assert (
        "Compound metadata field 'inchikey' does not match canonical_smiles"
        in res.stderr
    )
    assert not paths["html"].exists()


def test_stage9_report_fails_when_compound_smiles_is_not_canonical(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    paths["compound"].write_text(
        '{"inchikey": "TEST", "canonical_smiles": "C(C)O"}\n'
    )
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths)

    assert res.returncode != 0
    assert "Compound metadata field 'canonical_smiles' is not canonical" in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_fails_when_compound_input_type_missing(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    payload = json.loads(paths["compound"].read_text())
    del payload["input_type"]
    paths["compound"].write_text(json.dumps(payload) + "\n")
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths)

    assert res.returncode != 0
    assert "Compound metadata missing non-empty string field 'input_type'" in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_fails_when_input_canonical_smiles_mismatches(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    payload = json.loads(paths["compound"].read_text())
    payload["input_canonical_smiles"] = "C(C)O"
    paths["compound"].write_text(json.dumps(payload) + "\n")
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths)

    assert res.returncode != 0
    assert (
        "Compound metadata field 'input_canonical_smiles' does not match "
        "input_smiles"
    ) in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_fails_when_smiles_input_has_sdf_provenance(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    payload = json.loads(paths["compound"].read_text())
    payload["input_sdf"] = str(tmp_path / "stale.sdf")
    paths["compound"].write_text(json.dumps(payload) + "\n")
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths)

    assert res.returncode != 0
    assert (
        "Compound metadata field 'input_sdf' is not allowed when "
        "input_type is 'smiles'"
    ) in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_fails_when_sdf_input_has_smiles_provenance(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    payload = json.loads(paths["compound"].read_text())
    payload["input_type"] = "sdf"
    payload["input_sdf"] = str(tmp_path / "source_ligand.sdf")
    paths["compound"].write_text(json.dumps(payload) + "\n")
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths)

    assert res.returncode != 0
    assert (
        "Compound metadata field 'input_smiles' is not allowed when "
        "input_type is 'sdf'"
    ) in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_fails_when_skin_sens_decision_invalid(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    paths["admet"].write_text(
        '{"skin_sens": {"decision": "UNKNOWN"}, "structural_alerts": {}}\n'
    )
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths)

    assert res.returncode != 0
    assert "ADMET report skin_sens.decision must be one of" in res.stderr
    assert "UNKNOWN" in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_fails_when_skin_sens_decision_file_mismatch(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    paths["skin_sens_decision"].write_text("HALT\n")
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths)

    assert res.returncode != 0
    assert "Skin-sens decision file does not match ADMET report" in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_fails_when_structural_alerts_missing(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    paths["admet"].write_text('{"skin_sens": {"decision": "PASS"}}\n')
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths)

    assert res.returncode != 0
    assert "ADMET report missing object field 'structural_alerts'" in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_fails_when_skin_reaction_missing(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    report = json.loads(paths["admet"].read_text())
    del report["admet_ai_predictions"]["Skin_Reaction"]
    paths["admet"].write_text(json.dumps(report) + "\n")
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths)

    assert res.returncode != 0
    assert "ADMET report missing ADMET-AI endpoints" in res.stderr
    assert "Skin_Reaction" in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_fails_when_admet_source_report_mismatch(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    source = json.loads(paths["admet_ai"].read_text())
    source["predictions"]["DILI"] = 0.77
    paths["admet_ai"].write_text(json.dumps(source) + "\n")
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths)

    assert res.returncode != 0
    assert "source/report mismatch for endpoint DILI" in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_fails_when_admet_source_smiles_mismatch(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    source = json.loads(paths["admet_ai"].read_text())
    source["smiles"] = "CCC"
    paths["admet_ai"].write_text(json.dumps(source) + "\n")
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths)

    assert res.returncode != 0
    assert (
        "ADMET-AI source predictions smiles does not match compound canonical_smiles"
        in res.stderr
    )
    assert not paths["html"].exists()


def test_stage9_report_fails_when_skin_sens_source_report_mismatch(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    source = json.loads(paths["husspred"].read_text())
    source["skin_sens_call"] = "positive"
    source["skin_sens_probability"] = 0.8
    paths["husspred"].write_text(json.dumps(source) + "\n")
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths)

    assert res.returncode != 0
    assert "source/report mismatch for husspred call" in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_fails_when_skin_sens_report_is_degraded_by_default(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    report = json.loads(paths["admet"].read_text())
    report["skin_sens"]["degraded"] = True
    paths["admet"].write_text(json.dumps(report) + "\n")
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths)

    assert res.returncode != 0
    assert "ADMET report has degraded skin-sens evidence" in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_fails_when_structural_alert_source_report_mismatch(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    source = json.loads(paths["structural_alerts"].read_text())
    source["any_nih"] = True
    source["matches"]["NIH"] = ["nitro_aromatic"]
    paths["structural_alerts"].write_text(json.dumps(source) + "\n")
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths)

    assert res.returncode != 0
    assert "source/report mismatch for structural-alert flag any_nih" in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_fails_when_required_table_has_blank_target_id(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    paths["stage3"].write_text(
        "target_id,final_score,skin_score,source_count,sources,efficacy_top1\n"
        " ,1.0,0.8,3,autodock;gnina;rtmscore,hydration (12 papers)\n"
    )
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths)

    assert res.returncode != 0
    assert "Stage 3 ranked targets column 'target_id' contains blank values" in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_fails_when_required_table_has_duplicate_target_id(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    paths["stage3"].write_text(
        "target_id,final_score,skin_score,source_count,sources,efficacy_top1\n"
        "P1,1.0,0.8,3,autodock;gnina;rtmscore,hydration (12 papers)\n"
        "P1,0.9,0.7,3,autodock;gnina;rtmscore,barrier (4 papers)\n"
    )
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths)

    assert res.returncode != 0
    assert (
        "Stage 3 ranked targets contains duplicate target_id values: P1"
        in res.stderr
    )
    assert not paths["html"].exists()


def test_stage9_report_fails_when_stage3_source_count_mismatch(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    paths["stage3"].write_text(
        "target_id,final_score,skin_score,source_count,sources,efficacy_top1\n"
        "P1,1.0,0.8,3,autodock;gnina,hydration (12 papers)\n"
    )
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths)

    assert res.returncode != 0
    assert "source_count=3 but sources lists 2 label(s)" in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_fails_when_stage3_final_score_unsorted(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    paths["stage3"].write_text(
        "target_id,final_score,skin_score,source_count,sources,efficacy_top1\n"
        "P1,0.8,0.8,3,autodock;gnina;rtmscore,hydration (12 papers)\n"
        "P2,0.9,0.7,3,autodock;gnina;rtmscore,barrier (4 papers)\n"
    )
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths)

    assert res.returncode != 0
    assert "Stage 3 ranked targets final_score must be sorted descending" in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_fails_when_docking_rrf_is_nonnumeric(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    paths["stage3"].write_text(
        "target_id,final_score,skin_score,skin_tier,source_count,sources,docking_rrf,efficacy_top1\n"
        "P1,1.0,0.8,high,3,autodock;gnina;rtmscore,bad,hydration (12 papers)\n"
    )
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths)

    assert res.returncode != 0
    assert "Stage 3 ranked targets value 'docking_rrf' must be numeric" in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_fails_when_stage3_efficacy_columns_missing(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    paths["stage3"].write_text(
        "target_id,final_score,skin_score,source_count,sources\n"
        "P1,1.0,0.8,3,autodock;gnina;rtmscore\n"
    )
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths)

    assert res.returncode != 0
    assert "Stage 3 ranked targets missing efficacy_top* columns" in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_fails_when_mmgbsa_status_invalid(tmp_path: Path) -> None:
    paths = write_required_inputs(tmp_path)
    paths["mmgbsa"].write_text(
        "target_id\tmmgbsa_dg_kcal_mol\tstatus\nP1\t-12.3\tskipped\n"
    )
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths)

    assert res.returncode != 0
    assert "MM-GBSA report column 'status' contains invalid values" in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_fails_when_boltz_kept_invalid(tmp_path: Path) -> None:
    paths = write_required_inputs(tmp_path)
    paths["boltz"].write_text("target_id\tkept\nP1\tmaybe\n")
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths)

    assert res.returncode != 0
    assert "Boltz-2 cofold report column 'kept' contains invalid values" in res.stderr
    assert "maybe" in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_fails_when_dft_energy_nonnumeric(tmp_path: Path) -> None:
    paths = write_required_inputs(tmp_path)
    paths["dft"].write_text(
        "target_id\tdft_energy_hartree\tstatus\nP1\tbad\tok\n"
    )
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths)

    assert res.returncode != 0
    assert "DFT report column 'dft_energy_hartree' must be numeric" in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_fails_when_dft_energy_nonfinite(tmp_path: Path) -> None:
    paths = write_required_inputs(tmp_path)
    paths["dft"].write_text(
        "target_id\tdft_energy_hartree\tstatus\nP1\tinf\tok\n"
    )
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths)

    assert res.returncode != 0
    assert "DFT report column 'dft_energy_hartree' must be finite" in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_fails_when_dft_energy_nonnegative(tmp_path: Path) -> None:
    paths = write_required_inputs(tmp_path)
    paths["dft"].write_text(
        "target_id\tdft_energy_hartree\tstatus\nP1\t0\tok\nP2\t1\tok\n"
    )
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths)

    assert res.returncode != 0
    assert "DFT report column 'dft_energy_hartree' must be < 0" in res.stderr
    assert "row index(es) 0,1" in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_fails_when_ensemble_consensus_nonpositive(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    paths["ensemble"].write_text("target_id\tconsensus_score\nP1\t0\nP2\t-1\n")
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths)

    assert res.returncode != 0
    assert "Ensemble dock consensus column 'consensus_score' must be > 0" in res.stderr
    assert "row index(es) 0,1" in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_fails_when_mmgbsa_energy_nonfinite(tmp_path: Path) -> None:
    paths = write_required_inputs(tmp_path)
    paths["mmgbsa"].write_text(
        "target_id\tmmgbsa_dg_kcal_mol\tstatus\nP1\tinf\tok\n"
    )
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths)

    assert res.returncode != 0
    assert "MM-GBSA report column 'mmgbsa_dg_kcal_mol' must be finite" in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_accepts_nonnegative_mmgbsa_with_selection_reason(
    tmp_path: Path,
) -> None:
    """0/positive ΔTOTAL are valid computed results, not calculation errors."""
    paths = write_required_inputs(tmp_path)
    paths["mmgbsa"].write_text(
        "target_id\tmmgbsa_dg_kcal_mol\tstatus\nP1\t0\tok\n"
    )
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths)

    assert res.returncode == 0, res.stderr
    html = paths["html"].read_text(encoding="utf-8")
    assert "not_selected_nonnegative_delta_total" in html
    assert "selected_for_qm" in html


def test_stage9_report_fails_when_downstream_target_not_in_stage3(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    paths["dft"].write_text(
        "target_id\tdft_energy_hartree\tstatus\nP2\t-123.4\tok\n"
    )
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths)

    assert res.returncode != 0
    assert "DFT report contains target_id values absent from Stage 3 ranked targets" in res.stderr
    assert "P2" in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_fails_when_optional_json_invalid(tmp_path: Path) -> None:
    paths = write_required_inputs(tmp_path)
    cosing = tmp_path / "cosing.json"
    cosing.write_text("{not-json")
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths, ["--cosing-json", str(cosing)])

    assert res.returncode != 0
    assert "CosIng annotation is not valid JSON" in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_fails_when_optional_json_empty(tmp_path: Path) -> None:
    paths = write_required_inputs(tmp_path)
    warnings = tmp_path / "drug_warnings.json"
    warnings.write_text("")
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths, ["--drug-warnings-json", str(warnings)])

    assert res.returncode != 0
    assert "Drug-avoidance warnings is required and must be non-empty" in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_fails_when_disagreement_path_missing(tmp_path: Path) -> None:
    paths = write_required_inputs(tmp_path)
    missing = tmp_path / "missing_disagreement.json"
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths, ["--disagreement", str(missing)])

    assert res.returncode != 0
    assert "Disagreement analysis is required when its path is provided" in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_fails_when_cosing_tanimoto_missing(tmp_path: Path) -> None:
    paths = write_required_inputs(tmp_path)
    cosing = tmp_path / "cosing.json"
    cosing.write_text(
        '{"level": "SIMILAR", "inci": "Reference", '
        '"functions": ["Skin Conditioning"], "reference_status": "ok"}\n'
    )
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths, ["--cosing-json", str(cosing)])

    assert res.returncode != 0
    assert "CosIng annotation missing required metric 'tanimoto'" in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_fails_when_drug_max_tanimoto_missing(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    warnings = tmp_path / "drug_warnings.json"
    warnings.write_text('{"warnings": [], "n_warnings": 0, "reference_status": "ok"}\n')
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths, ["--drug-warnings-json", str(warnings)])

    assert res.returncode != 0
    assert (
        "Drug-avoidance warnings missing required metric "
        "'max_tanimoto_to_approved_drug'"
    ) in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_fails_when_optional_csv_empty(tmp_path: Path) -> None:
    paths = write_required_inputs(tmp_path)
    analogs = tmp_path / "analogs.csv"
    analogs.write_text("")
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths, ["--analogs-csv", str(analogs)])

    assert res.returncode != 0
    assert "Generated analogs table is empty" in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_fails_when_optional_csv_has_no_rows(tmp_path: Path) -> None:
    paths = write_required_inputs(tmp_path)
    synthesis = tmp_path / "synthesis.csv"
    synthesis.write_text("analog_id,smiles,best_route_score\n")
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths, ["--synthesis-csv", str(synthesis)])

    assert res.returncode != 0
    assert "Retrosynthesis ranking contains no rows" in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_fails_when_optional_analog_smiles_invalid(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    analogs = tmp_path / "analogs.csv"
    analogs.write_text("smiles\nnot-a-smiles\n")
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths, ["--analogs-csv", str(analogs)])

    assert res.returncode != 0
    assert "Generated analogs table column 'smiles' contains invalid SMILES" in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_fails_when_optional_synthesis_missing_id_column(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    synthesis = tmp_path / "synthesis.csv"
    synthesis.write_text("smiles,best_route_score\nCCO,1.0\n")
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths, ["--synthesis-csv", str(synthesis)])

    assert res.returncode != 0
    assert "Retrosynthesis ranking missing required columns" in res.stderr
    assert "analog_id" in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_fails_when_optional_synthesis_smiles_invalid(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    synthesis = tmp_path / "synthesis.csv"
    synthesis.write_text("analog_id,smiles,best_route_score\nA1,not-a-smiles,1.0\n")
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths, ["--synthesis-csv", str(synthesis)])

    assert res.returncode != 0
    assert "Retrosynthesis ranking column 'smiles' contains invalid SMILES" in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_fails_when_optional_kg_has_blank_target_id(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    kg_efficacy = tmp_path / "kg_efficacy.csv"
    kg_efficacy.write_text("target_id,efficacy_top1\n ,hydration\n")
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths, ["--kg-efficacy-csv", str(kg_efficacy)])

    assert res.returncode != 0
    assert "Skin-efficacy table column 'target_id' contains blank values" in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_fails_when_optional_kg_has_duplicate_target_id(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    kg_efficacy = tmp_path / "kg_efficacy.csv"
    kg_efficacy.write_text("target_id,efficacy_top1\nP1,hydration\nP1,barrier\n")
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths, ["--kg-efficacy-csv", str(kg_efficacy)])

    assert res.returncode != 0
    assert (
        "Skin-efficacy table contains duplicate target_id values: P1"
        in res.stderr
    )
    assert not paths["html"].exists()


def test_stage9_report_fails_when_target_metadata_lacks_labels(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    metadata = tmp_path / "target_metadata.tsv"
    metadata.write_text("target_id\nP1\n")
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths, ["--target-metadata", str(metadata)])

    assert res.returncode != 0
    assert "target metadata must include a gene or protein-name column" in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_writes_when_required_sources_valid(tmp_path: Path) -> None:
    paths = write_required_inputs(tmp_path)
    paths["stage3"].write_text(
        "target_id,final_score,skin_score,skin_tier,cell_type_preferred,"
        "source_count,sources,efficacy_top1\n"
        "P1,1.0,0.8,high,keratinocytes,3,autodock;gnina;rtmscore,"
        "hydration (12 papers)\n"
    )
    cosing = tmp_path / "cosing.json"
    cosing.write_text(
        '{"level": "SIMILAR", "inci": "Reference", '
        '"functions": ["Skin Conditioning"], "tanimoto": 0.9, '
        '"reference_status": "ok"}\n'
    )
    warnings = tmp_path / "drug_warnings.json"
    warnings.write_text(
        '{"max_tanimoto_to_approved_drug": 0.1, "warnings": [], '
        '"n_warnings": 0, "reference_status": "ok"}\n'
    )
    metadata = tmp_path / "target_metadata.tsv"
    metadata.write_text("Uniprot\tGene\tGene description\nP1\tGENE1\tProtein one\n")

    res = run_stage9(
        tmp_path,
        paths,
        [
            "--cosing-json",
            str(cosing),
            "--drug-warnings-json",
            str(warnings),
            "--target-metadata",
            str(metadata),
        ],
    )

    assert res.returncode == 0, res.stderr
    html = paths["html"].read_text()
    assert "Panel 0" in html
    assert "Report Decision" in html
    assert "Input type: <code>smiles</code>" in html
    assert "Input SMILES: <code>C(C)O</code>" in html
    assert "Input canonical SMILES: <code>CCO</code>" in html
    assert "InChIKey: <code>LFQSCWFLJHTTHZ-UHFFFAOYSA-N</code>" in html
    assert "canonical SMILES: <code>CCO</code>" in html
    assert "Overall decision: <span class='badge pass'>PASS</span>" in html
    assert "Recommended action: <strong>proceed</strong>" in html
    assert "Claimable: <strong>yes</strong>" in html
    assert "Claim status: proceed" in html
    assert "Efficacy claim allowed: <strong>no</strong>" in html
    assert "Skin context decision: <strong>skin_context_supported</strong>" in html
    assert "Skin context supported: <strong>yes</strong>" in html
    assert "Skin efficacy supported: <strong>yes</strong>" in html
    assert "Skin efficacy requires review: <strong>yes</strong>" in html
    assert (
        "skin efficacy evidence is automatic literature co-occurrence "
        "and requires review"
    ) in html
    assert "Skin_Reaction: <code>0.210 (low)</code>" in html
    assert "Cosmetic/drug decision: <strong>PROCEED</strong>" in html
    assert "Drug policy: <code>moderate</code>" in html
    assert "top target gene: <code>GENE1</code>" in html
    assert "top target protein: <code>Protein one</code>" in html
    assert "<th>gene_symbol</th>" in html
    assert "<td>GENE1</td>" in html
    assert "<th>protein_name</th>" in html
    assert "<td>Protein one</td>" in html
    assert "Structural alerts: <code>none</code>" in html
    assert "degraded skin-sens evidence: <code>False</code>" in html
    assert "missing skin-sens models: <code>none</code>" in html
    assert "Skin-sens 3-model evidence" in html
    assert "<th>model</th>" in html
    assert "<th>probability</th>" in html
    assert "<td>husspred</td>" in html
    assert "<td>stoptox</td>" in html
    assert "<td>pred_skin</td>" in html
    assert "<td>negative</td>" in html
    assert "<td>0.200</td>" in html
    assert "High ADMET risk endpoints: <code>none</code>" in html
    assert "Moderate ADMET risk endpoints: <code>none</code>" in html
    assert "Summary ADMET metrics" in html
    assert "<td>AMES</td>" in html
    assert "<td>0.04</td>" in html
    assert "<td>LD50_Zhu</td>" in html
    assert "<td>1.1</td>" in html
    assert "<td>tpsa</td>" in html
    assert "<td>20.23</td>" in html
    assert "Drug-avoidance warnings: <code>0</code>" in html
    assert "Screened target candidates: <code>1</code>" in html
    assert "Screening stage counts" in html
    assert "<td>autodock_screened_targets</td>" in html
    assert "<td>skin_weighted_ranked_targets</td>" in html
    assert "top target final score: 1.000" in html
    assert "top target source count: <code>3</code>" in html
    assert "top target sources: <code>autodock, gnina, rtmscore</code>" in html
    assert "top target HPA cell context: <code>keratinocytes</code>" in html
    assert "top target skin efficacy: <code>hydration (12 papers)</code>" in html
    assert "top target skin context supported: <strong>yes</strong>" in html
    assert "most skin-relevant HPA cell context: <code>keratinocytes</code>" in html
    assert "HPA cell context is an expression label" in html
    assert "Panel A" in html
    assert "ranked targets" in html
    assert "Skin Toxicity" in html
    assert "Skin_Reaction: 0.210" in html
    assert "skin sensitization and Skin_Reaction ADMET risk are low" in html
    assert "CosIng level: <strong>SIMILAR</strong>" in html
    assert "INCI = <code>Reference</code>" in html
    assert "functions = <code>Skin Conditioning</code>" in html
    assert "Tanimoto = 0.900" in html
    assert "max Tanimoto vs approved = 0.100" in html
    assert "Panel I" in html
    assert "DFT" in html


def test_stage9_report_requires_review_for_automatic_literature_efficacy(
    tmp_path: Path,
) -> None:
    """A PASS here contradicted the same panel's "review required" label.

    The only efficacy evidence in this fixture is the automatic gene-keyword
    co-occurrence column, so the report must stay non-claimable.
    """
    paths = write_required_inputs(tmp_path)

    res = run_stage9(tmp_path, paths)

    assert res.returncode == 0, res.stderr
    html = paths["html"].read_text()
    assert "Overall decision: <span class='badge pass'>PASS</span>" in html
    assert "Recommended action: <strong>proceed</strong>" in html
    assert "Claimable: <strong>yes</strong>" in html
    assert "Skin context supported: <strong>yes</strong>" in html
    assert "Skin efficacy supported: <strong>yes</strong>" in html
    assert "Skin efficacy requires review: <strong>yes</strong>" in html
    assert "Efficacy claim allowed: <strong>no</strong>" in html
    assert (
        "skin efficacy evidence is automatic literature co-occurrence "
        "and requires review"
    ) in html


def test_summary_overall_decision_requires_review_for_literature_efficacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import summarize_run_outputs

    overall = summarize_run_outputs._overall_decision(
        preset="target-id",
        safety={
            "skin_sens_decision": "PASS",
            "degraded": False,
            "missing_models": [],
            "structural_alert_flags": {},
            "admet_risk_assessment": {"high_risk_endpoints": [], "endpoints": {}},
        },
        skin_toxicity={"decision": "PASS"},
        cosmetic_drug={"decision": "PROCEED", "n_warnings": 0},
        target_prediction={
            "n_targets": 1,
            "top_targets": [{"target_id": "P1", "gene_symbol": "TYR"}],
        },
        skin_specialized_binding={
            "skin_context_supported": True,
            "skin_context_decision": "skin_context_supported",
            "top_target_skin_context_supported": True,
            "skin_efficacy_requires_review": True,
        },
    )

    # Operational completion stays PASS; the automatic literature signal only
    # removes the efficacy claim permission (D01 separation).
    assert overall["decision"] == "PASS"
    assert overall["requires_human_review"] is False
    assert overall["claimable"] is True
    assert overall["efficacy_claim_allowed"] is False
    assert (
        "skin efficacy evidence is automatic literature co-occurrence "
        "and requires review"
    ) in overall["reasons"]


def test_stage9_report_writes_sdf_input_provenance(tmp_path: Path) -> None:
    paths = write_required_inputs(tmp_path)
    payload = json.loads(paths["compound"].read_text())
    payload["input_type"] = "sdf"
    payload["input_sdf"] = str(tmp_path / "source_ligand.sdf")
    del payload["input_smiles"]
    del payload["input_canonical_smiles"]
    paths["compound"].write_text(json.dumps(payload) + "\n")

    res = run_stage9(tmp_path, paths)

    assert res.returncode == 0, res.stderr
    html = paths["html"].read_text()
    assert "Input type: <code>sdf</code>" in html
    assert f"Input SDF: <code>{tmp_path / 'source_ligand.sdf'}</code>" in html
    assert "Input SMILES:" not in html
    assert "Input canonical SMILES:" not in html
    assert "canonical SMILES: <code>CCO</code>" in html


def test_stage9_report_flags_top_binding_target_without_skin_context(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    paths["stage3"].write_text(
        "target_id,final_score,skin_score,skin_tier,source_count,sources,efficacy_top1\n"
        "P1,1.0,0.0,very_low,3,autodock;gnina;rtmscore,hydration (12 papers)\n"
        "P2,0.9,0.8,high,3,autodock;gnina;rtmscore,barrier (8 papers)\n"
    )

    res = run_stage9(tmp_path, paths)

    assert res.returncode == 0, res.stderr
    html = paths["html"].read_text()
    assert "Overall decision: <span class='badge flag'>FLAG_HIGH</span>" in html
    assert "Recommended action: <strong>review_before_claim</strong>" in html
    assert "Claimable: <strong>no</strong>" in html
    assert "Skin context decision: <strong>skin_context_supported</strong>" in html
    assert "Skin context supported: <strong>yes</strong>" in html
    assert "top target skin context supported: <strong>no</strong>" in html
    assert "top binding target lacks direct skin context support: P1" in html


def test_stage9_report_flags_high_admet_risk_for_review(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    predictions = dict(ADMET_PREDICTIONS)
    predictions["DILI"] = 0.72
    write_admet_bundle(paths, predictions=predictions)

    res = run_stage9(tmp_path, paths)

    assert res.returncode == 0, res.stderr
    html = paths["html"].read_text()
    assert "Overall decision: <span class='badge flag'>FLAG_HIGH</span>" in html
    assert "Recommended action: <strong>review_before_claim</strong>" in html
    assert "Claimable: <strong>no</strong>" in html
    assert "High ADMET risk endpoints: <code>DILI</code>" in html
    assert "<td>DILI</td>" in html
    assert "<td>0.72</td>" in html
    assert "high ADMET risk endpoints: DILI" in html


def test_stage9_report_flags_drug_warnings_for_review(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    warnings = tmp_path / "drug_warnings.json"
    warnings.write_text(
        '{"max_tanimoto_to_approved_drug": 0.7, '
        '"warnings": [{"tier": "SOFT_WARNING"}], '
        '"n_warnings": 1, "reference_status": "ok"}\n'
    )

    res = run_stage9(tmp_path, paths, ["--drug-warnings-json", str(warnings)])

    assert res.returncode == 0, res.stderr
    html = paths["html"].read_text()
    assert "Overall decision: <span class='badge flag'>FLAG_HIGH</span>" in html
    assert "Recommended action: <strong>review_before_claim</strong>" in html
    assert "Claimable: <strong>no</strong>" in html
    assert "Drug-avoidance warnings: <code>1</code>" in html
    assert "drug-avoidance warnings present: 1" in html


def test_stage9_report_fails_when_cosmetic_drug_decision_source_mismatch(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    paths["drug_warnings"].write_text(
        '{"max_tanimoto_to_approved_drug": 0.8, '
        '"warnings": [{"tier": "STRICT_WARNING"}], '
        '"n_warnings": 1, "reference_status": "ok"}\n'
    )
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    paths["html"].write_text("stale\n")

    res = run_stage9(tmp_path, paths)

    assert res.returncode != 0
    assert (
        "cosmetic/drug decision PROCEED does not match expected DOWNWEIGHT "
        "for policy moderate"
    ) in res.stderr
    assert not paths["html"].exists()


def test_stage9_report_flags_unsupported_skin_context_for_review(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    paths["stage3"].write_text(
        "target_id,final_score,skin_score,skin_tier,source_count,sources,efficacy_top1\n"
        "P1,1.0,0.0,very_low,3,autodock;gnina;rtmscore,hydration (12 papers)\n"
    )

    res = run_stage9(tmp_path, paths)

    assert res.returncode == 0, res.stderr
    html = paths["html"].read_text()
    assert "Overall decision: <span class='badge flag'>FLAG_HIGH</span>" in html
    assert "Recommended action: <strong>review_before_claim</strong>" in html
    assert "Claimable: <strong>no</strong>" in html
    assert "Claim status: human review required before claim" in html
    assert "Skin context decision: <strong>skin_efficacy_literature_only</strong>" in html
    assert "Skin context supported: <strong>no</strong>" in html
    assert (
        "skin-specialized binding context requires review: "
        "skin_efficacy_literature_only"
    ) in html


def test_stage9_report_renders_moderate_skin_toxicity_review(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    predictions = dict(ADMET_PREDICTIONS)
    predictions["Skin_Reaction"] = 0.31
    write_admet_bundle(paths, predictions=predictions)

    res = run_stage9(tmp_path, paths)

    assert res.returncode == 0, res.stderr
    html = paths["html"].read_text()
    assert "Skin Toxicity" in html
    assert "Skin toxicity: <span class='badge flag'>REVIEW</span>" in html
    assert "toxicity level: <strong>moderate</strong>" in html
    assert "Skin_Reaction: 0.310 (moderate)" in html
    assert "Skin_Reaction ADMET risk is moderate" in html


def test_stage9_report_renders_skin_weighted_efficacy_targets(
    tmp_path: Path,
) -> None:
    paths = write_required_inputs(tmp_path)
    paths["stage3"].write_text(
        "target_id,final_score,skin_score,source_count,sources,efficacy_top1\n"
        "P1,0.91,0.82,3,autodock;gnina;rtmscore,hydration (12 papers)\n"
    )

    res = run_stage9(
        tmp_path,
        paths,
        ["--kg-efficacy-csv", str(paths["stage3"])],
    )

    assert res.returncode == 0, res.stderr
    html = paths["html"].read_text()
    assert "Panel A" in html
    assert "final_score" in html
    assert "skin_score" in html
    assert "Panel H" in html
    assert "hydration (12 papers)" in html


def test_halt_blocks_efficacy_claim_permission(monkeypatch) -> None:
    """전체 중단(HALT)이면 효능 주장 권한도 false여야 한다(F04)."""
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import summarize_run_outputs

    overall = summarize_run_outputs._overall_decision(
        preset="target-id",
        safety={
            "skin_sens_decision": "HALT",
            "degraded": False,
            "missing_models": [],
            "structural_alert_flags": {},
            "admet_risk_assessment": {"high_risk_endpoints": [], "endpoints": {}},
        },
        skin_toxicity={"decision": "HALT"},
        cosmetic_drug={"decision": "PROCEED", "n_warnings": 0},
        target_prediction={
            "n_targets": 1,
            "top_targets": [{"target_id": "P1", "gene_symbol": "TYR"}],
        },
        skin_specialized_binding={
            "skin_context_supported": True,
            "skin_context_decision": "skin_context_supported",
            "top_target_skin_context_supported": True,
            "skin_efficacy_requires_review": False,
        },
    )

    assert overall["decision"] == "HALT"
    assert overall["claimable"] is False
    assert overall["efficacy_claim_allowed"] is False
