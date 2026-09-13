from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "stage0_download_alphafold.sh"


def _offline_env(tmp_path: Path) -> dict[str, str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for name, body in {
        "df": "printf 'Filesystem 1K-blocks Used Available Use%% Mounted\\nfixture 2000000000 1 1000000000 1%% /\\n'\n",
        "wget": "echo 'unexpected download in offline regression' >&2\nexit 97\n",
    }.items():
        script = fake_bin / name
        script.write_text("#!/bin/sh\n" + body)
        script.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env.get('PATH', '')}"
    return env


def test_reuses_validated_existing_extraction(tmp_path: Path) -> None:
    out_dir = tmp_path / "alphafold"
    out_dir.mkdir()
    (out_dir / ".extracted").touch()
    for index in range(3):
        (out_dir / f"AF-P{index}-F1-model_v4.pdb").write_text("MODEL\n", encoding="ascii")
        (out_dir / f"AF-P{index}-F1-model_v4.cif.gz").write_bytes(b"mmCIF fixture")

    env = _offline_env(tmp_path)
    env["MIN_EXISTING_MODELS"] = "3"
    result = subprocess.run(
        ["bash", str(SCRIPT), "https://invalid.example/unused.tar", str(out_dir)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Reusing extracted AlphaFold proteome (3 PDB and 3 canonical mmCIF files)." in result.stdout
    assert not (out_dir / "unused.tar").exists()


def test_pdb_only_extraction_is_invalidated_for_canonical_sequence_recovery(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "alphafold"
    out_dir.mkdir()
    (out_dir / ".extracted").touch()
    for index in range(3):
        (out_dir / f"AF-P{index}-F1-model_v4.pdb").write_text(
            "MODEL\n", encoding="ascii"
        )
    # A retained archive lets the downloader hand control to the extraction
    # rule without network access; the stale marker must no longer short-circuit it.
    (out_dir / "proteome.tar").write_bytes(b"fixture")
    env = _offline_env(tmp_path)
    env["MIN_EXISTING_MODELS"] = "3"
    env["MIN_TAR_GB"] = "0"

    result = subprocess.run(
        ["bash", str(SCRIPT), "https://invalid.example/proteome.tar", str(out_dir)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "3 PDB, 0 canonical mmCIF" in result.stdout
    assert not (out_dir / ".extracted").exists()
