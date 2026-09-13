"""Unit tests for the §13.2 AiZynthFinder runner fail-closed behavior."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "stage7_5_aizynth.py"


def write_sdf(path: Path, smiles: str = "CCO") -> None:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()


def write_fake_aizynth(fake_bin: Path, body: str) -> None:
    fake_bin.mkdir()
    exe = fake_bin / "aizynthcli"
    exe.write_text(f"#!{sys.executable}\n{body}")
    exe.chmod(0o755)


def test_cli_rejects_missing_aizynth_binary(tmp_path: Path) -> None:
    in_sdf = tmp_path / "analogs.sdf"
    write_sdf(in_sdf)
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("stale\n")
    env = os.environ.copy()
    env["PATH"] = ""

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--in-sdf", str(in_sdf),
         "--out-dir", str(tmp_path / "routes"),
         "--out-manifest", str(manifest)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode != 0
    assert "aizynthcli is not available on PATH" in result.stderr
    assert not manifest.exists()


def test_cli_rejects_missing_input_sdf_without_stale_manifest(tmp_path: Path) -> None:
    in_sdf = tmp_path / "missing.sdf"
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("stale\n")
    env = os.environ.copy()
    env["PATH"] = ""

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--in-sdf", str(in_sdf),
         "--out-dir", str(tmp_path / "routes"),
         "--out-manifest", str(manifest)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode != 0
    assert "AiZynth input SDF is required and must be non-empty" in result.stderr
    assert not manifest.exists()


def test_cli_rejects_empty_input_sdf_without_stale_manifest(tmp_path: Path) -> None:
    in_sdf = tmp_path / "analogs.sdf"
    in_sdf.write_text("")
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("stale\n")
    env = os.environ.copy()
    env["PATH"] = ""

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--in-sdf", str(in_sdf),
         "--out-dir", str(tmp_path / "routes"),
         "--out-manifest", str(manifest)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode != 0
    assert "AiZynth input SDF is required and must be non-empty" in result.stderr
    assert not manifest.exists()


def test_cli_writes_manifest_for_valid_zero_route_payload(tmp_path: Path) -> None:
    in_sdf = tmp_path / "analogs.sdf"
    write_sdf(in_sdf)
    manifest = tmp_path / "manifest.tsv"
    fake_bin = tmp_path / "bin"
    write_fake_aizynth(
        fake_bin,
        """
import json
import sys

out = sys.argv[sys.argv.index("--output") + 1]
with open(out, "w") as fh:
    json.dump({"routes": []}, fh)
""",
    )
    env = os.environ.copy()
    env["PATH"] = str(fake_bin)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--in-sdf", str(in_sdf),
         "--out-dir", str(tmp_path / "routes"),
         "--out-manifest", str(manifest)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    rows = manifest.read_text().splitlines()
    assert rows[0] == "analog_id\tsmiles\troutes_json\tn_routes"
    assert rows[1].startswith("analog_0001\tCCO\t")
    assert rows[1].endswith("\t0")


def test_cli_rejects_success_without_route_json(tmp_path: Path) -> None:
    in_sdf = tmp_path / "analogs.sdf"
    write_sdf(in_sdf)
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("stale\n")
    stale_route = tmp_path / "routes" / "analog_0001" / "routes.json"
    stale_route.parent.mkdir(parents=True)
    stale_route.write_text('{"routes": [{"n_steps": 1}]}\n')
    fake_bin = tmp_path / "bin"
    write_fake_aizynth(fake_bin, "import sys\nsys.exit(0)\n")
    env = os.environ.copy()
    env["PATH"] = str(fake_bin)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--in-sdf", str(in_sdf),
         "--out-dir", str(tmp_path / "routes"),
         "--out-manifest", str(manifest)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode != 0
    assert "did not produce routes JSON" in result.stderr
    assert not manifest.exists()
    assert not stale_route.exists()
