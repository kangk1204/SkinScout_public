"""Alignment-identity sequence leakage contract (audit F10).

The sequence leakage axis must use MMseqs alignment identity plus alignment
coverage.  There is deliberately no LCS/containment fallback: when MMseqs is
unavailable (or a record cannot produce a covered alignment) the axis is
INCOMPLETE, never silently replaced by another similarity statistic.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))

import leakage_check  # noqa: E402
from leakage_check import Thresholds, audit, mmseqs_max_seq_id  # noqa: E402

QUERY = "M" + "ACDEFGHIKLMNPQRSTVWY" * 2


def _install_fake_mmseqs(bin_dir: Path, output: str) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    script = bin_dir / "mmseqs"
    script.write_text(
        "#!/bin/sh\n"
        "test \"$1\" = easy-search || exit 2\n"
        f"printf '{output}' > \"$4\"\n"
    )
    script.chmod(0o755)


def _reference(tmp_path: Path, sequence: str) -> Path:
    fasta = tmp_path / "training_cutoff_seqs.fasta"
    fasta.write_text(f">ref\n{sequence}\n")
    return fasta


def _search_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output: str) -> Path:
    _install_fake_mmseqs(tmp_path / "bin", output)
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.defpath}")
    return _reference(tmp_path, QUERY)


def test_independent_sequence_is_negative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fasta = _search_path(tmp_path, monkeypatch, "")

    assert mmseqs_max_seq_id("P12345", fasta, query_sequence=QUERY) == 0.0


def test_real_homolog_is_positive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fasta = _search_path(
        tmp_path,
        monkeypatch,
        "P12345\\tref\\t92.0\\t40\\t0.98\\t0.95\\n",
    )

    score = mmseqs_max_seq_id("P12345", fasta, query_sequence=QUERY)

    assert score == 0.92
    assert score >= 0.30


def test_low_quality_micro_alignment_is_not_a_homolog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fasta = _search_path(
        tmp_path,
        monkeypatch,
        "P12345\\tref\\t100.0\\t4\\t0.10\\t0.05\\n",
    )

    assert mmseqs_max_seq_id("P12345", fasta, query_sequence=QUERY) == 0.0


def test_short_query_micro_alignment_is_not_a_homolog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fasta = _search_path(
        tmp_path,
        monkeypatch,
        "P12345\\tref\\t100.0\\t6\\t1.0\\t0.05\\n",
    )

    assert mmseqs_max_seq_id("P12345", fasta, query_sequence="MABCDE") == 0.0


def test_missing_mmseqs_is_incomplete_not_containment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(leakage_check.shutil, "which", lambda _: None)
    fasta = _reference(tmp_path, "X" * 20 + QUERY + "Y" * 20)

    assert mmseqs_max_seq_id("P12345", fasta, query_sequence=QUERY) is None


def test_alignment_identity_is_not_maxed_with_containment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    query = "MABCDE" * 4
    fasta = _reference(tmp_path, "X" * 20 + query + "Y" * 20)
    _install_fake_mmseqs(
        tmp_path / "bin",
        "P12345\\tref\\t10.0\\t24\\t1.0\\t0.375\\n",
    )
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.defpath}")

    score = mmseqs_max_seq_id("P12345", fasta, query_sequence=query)

    assert score == 0.10, "MMseqs identity must not be raised by containment"


def test_domain_alignment_covering_the_shorter_sequence_is_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fasta = _search_path(
        tmp_path,
        monkeypatch,
        "P12345\\tref\\t80.0\\t40\\t1.0\\t0.04\\n",
    )

    assert mmseqs_max_seq_id("P12345", fasta, query_sequence=QUERY) == 0.80


def test_audit_marks_sequence_incomplete_when_mmseqs_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(leakage_check.shutil, "which", lambda _: None)
    fasta = _reference(tmp_path, QUERY)
    ligands = tmp_path / "ligands.smi"
    ligands.write_text("c1ccccc1 ref\n")
    pockets = tmp_path / "pockets.tsv"
    pockets.write_text("target_id\tmax_sucos\nP12345\t0.1\n")
    rows = pd.DataFrame([{"uniprot": "P12345", "smiles": "CCO", "sequence": QUERY}])

    out = audit(
        rows,
        fasta,
        ligands,
        pockets,
        Thresholds(0.30, 0.50, 0.50),
        allow_incomplete=True,
    )

    assert out.loc[0, "audit_status"] == "incomplete"
    assert "sequence" in out.loc[0, "missing_axes"]
    assert not bool(out.loc[0, "leak_flag"])


def test_audit_flags_alignment_identity_homolog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fasta = _search_path(
        tmp_path,
        monkeypatch,
        "P12345\\tref\\t42.0\\t40\\t0.95\\t0.90\\n",
    )
    ligands = tmp_path / "ligands.smi"
    ligands.write_text("c1ccccc1 ref\n")
    pockets = tmp_path / "pockets.tsv"
    pockets.write_text("target_id\tmax_sucos\nP12345\t0.1\n")
    rows = pd.DataFrame([{"uniprot": "P12345", "smiles": "CCO", "sequence": QUERY}])

    out = audit(rows, fasta, ligands, pockets, Thresholds(0.30, 0.50, 0.50))

    assert out.loc[0, "seq_id"] == 0.42
    assert out.loc[0, "audit_status"] == "ok"
    assert bool(out.loc[0, "leak_flag"])


def test_no_lcs_helper_remains() -> None:
    assert not hasattr(leakage_check, "_sequence_identity")
