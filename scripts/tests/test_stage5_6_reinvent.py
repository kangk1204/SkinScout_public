"""Tests for the manifest-driven Stage 5.6 REINVENT4 adapter."""

from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path

from rdkit import Chem


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/stage5_6_reinvent.py"


def write_input_sdf(path: Path) -> None:
    molecule = Chem.MolFromSmiles("CCO")
    assert molecule is not None
    writer = Chem.SDWriter(str(path))
    writer.write(molecule)
    writer.close()


def write_fake_executable(path: Path, *, body: str | None = None) -> None:
    script = body or """#!/usr/bin/env python3
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--anchors", required=True)
parser.add_argument("--seed", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()
with open(args.out, "w") as handle:
    handle.write("smiles,name\\n")
    handle.write("CCO,ethanol\\n")
    handle.write("OCC,duplicate\\n")
    handle.write("CCN,ethylamine\\n")
print("CCCl")
print("fake stderr", file=sys.stderr)
"""
    path.write_text(script)
    path.chmod(path.stat().st_mode | 0o111)


def write_inputs(tmp_path: Path, *, missing_artifact: bool = False) -> dict[str, Path]:
    input_sdf = tmp_path / "ligand.sdf"
    write_input_sdf(input_sdf)
    consensus = tmp_path / "consensus.json"
    consensus.write_text(json.dumps({"P12345": {"confirmed_atoms": [0], "claim_eligible": True}}))
    boltz_report = tmp_path / "boltz_report.tsv"
    complex_pdb = tmp_path / "complex.pdb"
    complex_pdb.write_text(
        "HETATM    1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "END\n"
    )
    boltz_report.write_text(
        "target_id\tcomplex_pdb\tiptm\tcomplex_plddt\taffinity_log_uM\tkept\n"
        f"P12345\t{complex_pdb}\t0.8\t90.0\t-1.0\tyes\n"
    )
    interaction_anchors = tmp_path / "interaction_anchor_map.json"
    parent = next(m for m in Chem.SDMolSupplier(str(input_sdf)) if m is not None)
    source_records = {
        "parent_sdf": input_sdf,
        "consensus_json": consensus,
        "boltz_report": boltz_report,
    }
    interaction_anchors.write_text(
        json.dumps(
            {
                "schema_version": "skinscout.interaction_anchor_map.v1",
                "consensus_sha256": hashlib.sha256(consensus.read_bytes()).hexdigest(),
                "boltz_report_sha256": hashlib.sha256(boltz_report.read_bytes()).hexdigest(),
                "sources": {
                    name: {
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for name, path in source_records.items()
                },
                "parent": {
                    "sdf_sha256": hashlib.sha256(input_sdf.read_bytes()).hexdigest(),
                    "inchikey": Chem.MolToInchiKey(parent),
                    "atom_count": parent.GetNumAtoms(),
                },
                "targets": {
                    "P12345": {
                        "target_id": "P12345",
                        "source_coordinate_system": "boltz_complex_ligand_atom_order_0_based",
                        "mapped_coordinate_system": "parent_sdf_atom_order_0_based",
                        "confirmed_complex_atoms": [0],
                        "confirmed_parent_atoms": [0],
                        "complex_pdb": str(complex_pdb),
                        "complex_pdb_bytes": complex_pdb.stat().st_size,
                        "complex_pdb_sha256": hashlib.sha256(
                            complex_pdb.read_bytes()
                        ).hexdigest(),
                        "atom_mapping": [
                            {"complex_atom_index": 0, "parent_atom_index": 0}
                        ],
                        "mapping_status": "mapped",
                        "mapping_confidence": "high",
                        "claim_eligible": True,
                    }
                },
                "claim_eligible": True,
                "analog_pose_verified": False,
            }
        )
    )
    config = tmp_path / "reinvent_config.toml"
    config.write_text("[run]\nmode = 'fake'\n")
    model = tmp_path / "prior.chkpt"
    if not missing_artifact:
        model.write_text("model bytes\n")
    model_manifest = tmp_path / "model_manifest.json"
    model_manifest.write_text(json.dumps({"artifacts": [{"name": "prior", "path": model.name}]}))
    plugin_file = tmp_path / "plugin.py"
    plugin_file.write_text("# fake plugin\n")
    plugin_manifest = tmp_path / "plugin_manifest.json"
    plugin_manifest.write_text(
        json.dumps(
            {
                "name": "fake-reinvent4-plugin",
                "required_files": [plugin_file.name],
                "command_args": [
                    "--config",
                    "{config}",
                    "--model",
                    "{first_model_artifact}",
                    "--anchors",
                    "{interaction_anchors}",
                    "--seed",
                    "{seed}",
                    "--out",
                    "{generated_output}",
                ],
            }
        )
    )
    executable = tmp_path / "fake_reinvent.py"
    write_fake_executable(executable)
    return {
        "input_sdf": input_sdf,
        "consensus": consensus,
        "interaction_anchors": interaction_anchors,
        "boltz_report": boltz_report,
        "complex_pdb": complex_pdb,
        "config": config,
        "model_manifest": model_manifest,
        "plugin_manifest": plugin_manifest,
        "executable": executable,
    }


def run_adapter(tmp_path: Path, *, extra_args: list[str] | None = None, paths: dict[str, Path] | None = None) -> subprocess.CompletedProcess[str]:
    inputs = paths or write_inputs(tmp_path)
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--in-sdf",
            str(inputs["input_sdf"]),
            "--consensus",
            str(inputs["consensus"]),
            "--interaction-anchors",
            str(inputs["interaction_anchors"]),
            "--boltz-report",
            str(inputs["boltz_report"]),
            "--model-manifest",
            str(inputs["model_manifest"]),
            "--config",
            str(inputs["config"]),
            "--plugin-manifest",
            str(inputs["plugin_manifest"]),
            "--executable",
            str(inputs["executable"]),
            "--seed",
            "12345",
            "--out-smi",
            str(tmp_path / "generated.smi"),
            "--out-lineage-csv",
            str(tmp_path / "lineage.csv"),
            "--out-lineage-json",
            str(tmp_path / "lineage.json"),
            "--out-status",
            str(tmp_path / "status.json"),
            *(extra_args or []),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_success_normalizes_smiles_lineage_logs_and_status_hashes(tmp_path: Path) -> None:
    result = run_adapter(tmp_path)

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "generated.smi").read_text().splitlines() == [
        "CCO\treinvent4_000001",
        "CCN\treinvent4_000002",
        "CCCl\treinvent4_000003",
    ]
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["execution_status"] == "completed"
    assert status["claim_ready"] is False
    assert status["claim_eligible"] is False
    assert status["seed"] == 12345
    assert status["generated_count"] == 3
    assert status["interaction_anchor_targets"] == ["P12345"]
    assert status["interaction_anchor_sha256"]
    assert status["output_hashes"]["smi"]
    assert status["stdout_sha256"]
    assert status["stderr_sha256"]
    assert "fake stderr" in (tmp_path / "status.stderr.txt").read_text()
    lineage = json.loads((tmp_path / "lineage.json").read_text())
    assert [row["smiles"] for row in lineage["molecules"]] == ["CCO", "CCN", "CCCl"]


def test_missing_model_artifact_blocks_only_with_allow_empty_output(tmp_path: Path) -> None:
    paths = write_inputs(tmp_path, missing_artifact=True)

    result = run_adapter(tmp_path, paths=paths)

    assert result.returncode != 0
    assert "REINVENT model artifact prior is required" in result.stderr
    assert not (tmp_path / "status.json").exists()
    assert not (tmp_path / "generated.smi").exists()

    blocked = run_adapter(tmp_path, paths=paths, extra_args=["--allow-empty-output"])

    assert blocked.returncode == 0, blocked.stderr
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["execution_status"] == "blocked"
    assert status["claim_ready"] is False
    assert "REINVENT model artifact prior is required" in status["blocker"]
    assert (tmp_path / "generated.smi").read_text() == ""
    assert json.loads((tmp_path / "lineage.json").read_text())["molecules"] == []


def test_nonzero_exit_captures_no_stale_outputs_on_failure(tmp_path: Path) -> None:
    paths = write_inputs(tmp_path)
    write_fake_executable(
        paths["executable"],
        body="""#!/usr/bin/env python3
import sys
print("partial stdout")
print("boom", file=sys.stderr)
raise SystemExit(17)
""",
    )
    for name in ["status.json", "generated.smi", "lineage.csv", "lineage.json", "status.stdout.txt", "status.stderr.txt"]:
        (tmp_path / name).write_text("stale\n")

    result = run_adapter(tmp_path, paths=paths)

    assert result.returncode != 0
    assert "exit code 17" in result.stderr
    for name in ["status.json", "generated.smi", "lineage.csv", "lineage.json", "status.stdout.txt", "status.stderr.txt"]:
        assert not (tmp_path / name).exists()


def test_empty_output_is_failure_not_blocked_diagnostic(tmp_path: Path) -> None:
    paths = write_inputs(tmp_path)
    write_fake_executable(
        paths["executable"],
        body="""#!/usr/bin/env python3
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--config")
parser.add_argument("--model")
parser.add_argument("--anchors")
parser.add_argument("--seed")
parser.add_argument("--out")
parser.parse_args()
""",
    )

    result = run_adapter(tmp_path, paths=paths, extra_args=["--allow-empty-output"])

    assert result.returncode != 0
    assert "produced no generated SMILES" in result.stderr
    assert not (tmp_path / "status.json").exists()
    assert not (tmp_path / "generated.smi").exists()


def test_seed_must_be_declared_and_is_propagated_to_executable(tmp_path: Path) -> None:
    paths = write_inputs(tmp_path)
    seed_capture = tmp_path / "seen_seed.txt"
    write_fake_executable(
        paths["executable"],
        body=f"""#!/usr/bin/env python3
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--config")
parser.add_argument("--model")
parser.add_argument("--anchors")
parser.add_argument("--seed", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()
open({str(seed_capture)!r}, "w").write(args.seed)
open(args.out, "w").write("CCO\\n")
""",
    )

    result = run_adapter(tmp_path, paths=paths)

    assert result.returncode == 0, result.stderr
    assert seed_capture.read_text() == "12345"

    plugin_manifest = json.loads(paths["plugin_manifest"].read_text())
    plugin_manifest["command_args"] = ["--config", "{config}", "--out", "{generated_output}"]
    paths["plugin_manifest"].write_text(json.dumps(plugin_manifest))
    result_without_seed = run_adapter(tmp_path, paths=paths)

    assert result_without_seed.returncode != 0
    assert "must propagate the deterministic {seed}" in result_without_seed.stderr
    assert not (tmp_path / "status.json").exists()


def test_plugin_must_receive_target_conditioned_interaction_anchors(
    tmp_path: Path,
) -> None:
    paths = write_inputs(tmp_path)
    plugin_manifest = json.loads(paths["plugin_manifest"].read_text())
    plugin_manifest["command_args"] = [
        "--config",
        "{config}",
        "--model",
        "{model}",
        "--seed",
        "{seed}",
        "--out",
        "{generated_output}",
    ]
    paths["plugin_manifest"].write_text(json.dumps(plugin_manifest))

    result = run_adapter(tmp_path, paths=paths)

    assert result.returncode != 0
    assert "target-conditioned {interaction_anchors} path" in result.stderr
    assert not (tmp_path / "status.json").exists()


def test_interaction_anchor_sources_are_verified_before_generation(
    tmp_path: Path,
) -> None:
    paths = write_inputs(tmp_path)
    original = paths["complex_pdb"].read_bytes()
    paths["complex_pdb"].write_bytes(original[::-1])

    result = run_adapter(tmp_path, paths=paths)

    assert result.returncode != 0
    assert "complex PDB" in result.stderr
    assert "fingerprint-mismatched" in result.stderr
    assert not (tmp_path / "generated.smi").exists()
    assert not (tmp_path / "status.json").exists()


def test_output_hashes_match_written_outputs(tmp_path: Path) -> None:
    result = run_adapter(tmp_path)

    assert result.returncode == 0, result.stderr
    status = json.loads((tmp_path / "status.json").read_text())
    for label, filename in {
        "smi": "generated.smi",
        "lineage_csv": "lineage.csv",
        "lineage_json": "lineage.json",
        "stdout": "status.stdout.txt",
        "stderr": "status.stderr.txt",
    }.items():
        data = (tmp_path / filename).read_bytes()
        assert status["output_hashes"][label] == __import__("hashlib").sha256(data).hexdigest()
