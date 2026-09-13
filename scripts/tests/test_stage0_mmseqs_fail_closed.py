"""Regression tests for Stage 0 MMseqs leakage-reference gates."""

from __future__ import annotations

import os
import gzip
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def write_mmseqs_stub(tmp_path: Path, *, fail_createindex: bool = False) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "mmseqs"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "cmd=${1:-}\n"
        "if [ \"$cmd\" = \"createdb\" ]; then\n"
        "  touch \"$3.dbtype\" \"$3.lookup\"\n"
        "elif [ \"$cmd\" = \"createindex\" ]; then\n"
        f"  exit {9 if fail_createindex else 0}\n"
        "else\n"
        "  echo unknown mmseqs command: \"$cmd\" >&2\n"
        "  exit 2\n"
        "fi\n"
    )
    stub.chmod(0o755)
    return bin_dir


def write_canonical_fasta(path: Path) -> None:
    path.write_text(">sp|P12345|TEST_HUMAN\nAG\n")


def run_mmseqs_build(
    tmp_path: Path,
    *extra: str,
    with_training_cutoff: bool = False,
    with_fetch_sources: bool = False,
    with_broken_fetch_sources: bool = False,
    fail_createindex: bool = False,
) -> subprocess.CompletedProcess[str]:
    canonical_fasta = tmp_path / "human_canonical.fasta"
    out_dir = tmp_path / "mmseqs"
    write_canonical_fasta(canonical_fasta)
    if with_training_cutoff:
        out_dir.mkdir()
        (out_dir / "training_cutoff_seqs.fasta").write_text(">train\nAG\n")
    env = os.environ.copy()
    env["PATH"] = (
        f"{write_mmseqs_stub(tmp_path, fail_createindex=fail_createindex)}:"
        f"{env['PATH']}"
    )
    if with_fetch_sources:
        sources = tmp_path / "sources"
        sources.mkdir()
        entries = sources / "entries.idx"
        entries.write_text(
            "IDCODE, HEADER, ACCESSION DATE, COMPOUND, SOURCE, AUTHOR LIST, RESOLUTION, EXPERIMENT TYPE (IF NOT X-RAY)\n"
            "------- ------- --------------- --------- ------- ------------ ----------- ----------------------------------------------------------------\n"
            "1AAA\tHYDROLASE\t09/29/21\told protein\tHomo sapiens\tA.\t1.5\tX-RAY DIFFRACTION\n"
            "9ZZZ\tHYDROLASE\t10/01/21\tfuture protein\tHomo sapiens\tB.\t1.5\tX-RAY DIFFRACTION\n"
        )
        seqres = sources / "pdb_seqres.txt.gz"
        with gzip.open(seqres, "wt") as fh:
            fh.write(
                ">1aaa_A mol:protein length:2  OLD\n"
                "AG\n"
                ">9zzz_A mol:protein length:2  FUTURE\n"
                "AG\n"
                ">1aaa_B mol:na length:2  DNA\n"
                "AG\n"
            )
        env["RCSB_ENTRIES_IDX_URL"] = str(entries)
        env["RCSB_SEQRES_FASTA_URL"] = str(seqres)
        env["RCSB_MIN_TRAINING_CUTOFF_SEQUENCES"] = "1"
    if with_broken_fetch_sources:
        env["RCSB_ENTRIES_IDX_URL"] = str(tmp_path / "missing_entries.idx")
        env["RCSB_SEQRES_FASTA_URL"] = str(tmp_path / "missing_seqres.txt.gz")
        env["RCSB_MIN_TRAINING_CUTOFF_SEQUENCES"] = "1"
    return subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/stage0_mmseqs_build.sh"),
            str(canonical_fasta),
            str(out_dir),
            *extra,
        ],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )


def test_mmseqs_build_requires_training_cutoff_by_default(tmp_path: Path) -> None:
    res = run_mmseqs_build(tmp_path, with_broken_fetch_sources=True)

    assert res.returncode != 0
    assert "training cutoff FASTA" in res.stderr


def test_mmseqs_build_allows_missing_training_cutoff_only_for_diagnostics(tmp_path: Path) -> None:
    res = run_mmseqs_build(tmp_path, "--allow-missing-training-cutoff")

    assert res.returncode == 0, res.stderr
    assert "sequence leakage audit disabled" in res.stderr


def test_mmseqs_build_writes_training_db_when_cutoff_fasta_exists(tmp_path: Path) -> None:
    res = run_mmseqs_build(tmp_path, with_training_cutoff=True)

    assert res.returncode == 0, res.stderr
    assert (tmp_path / "mmseqs" / "training_cutoff_db.dbtype").exists()


def test_mmseqs_build_fails_when_index_creation_fails(tmp_path: Path) -> None:
    res = run_mmseqs_build(
        tmp_path,
        with_training_cutoff=True,
        fail_createindex=True,
    )

    assert res.returncode != 0
    assert "[stage0.mmseqs] done." not in res.stdout


def test_mmseqs_build_fetches_training_cutoff_when_missing(tmp_path: Path) -> None:
    res = run_mmseqs_build(tmp_path, with_fetch_sources=True)

    assert res.returncode == 0, res.stderr
    fasta = tmp_path / "mmseqs" / "training_cutoff_seqs.fasta"
    assert fasta.read_text() == ">1aaa_A mol:protein length:2  OLD\nAG\n"
    assert (tmp_path / "mmseqs" / "training_cutoff_manifest.json").exists()
    assert (tmp_path / "mmseqs" / "training_cutoff_db.dbtype").exists()


def test_training_cutoff_fetch_writes_manifest_for_existing_fasta(tmp_path: Path) -> None:
    out_dir = tmp_path / "mmseqs"
    out_dir.mkdir()
    fasta = out_dir / "training_cutoff_seqs.fasta"
    fasta.write_text(">1aaa_A mol:protein length:2 OLD\nAG\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage0_fetch_training_cutoff.py"),
            "--out-dir",
            str(out_dir),
            "--min-sequences",
            "1",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert res.returncode == 0, res.stderr
    manifest = (out_dir / "training_cutoff_manifest.json").read_text()
    assert '"protein_sequence_count": 1' in manifest
    assert '"reused_existing_fasta": true' in manifest
