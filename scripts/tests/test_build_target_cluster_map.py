"""Regression tests for auditable MMseqs2 target cluster maps."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "eval" / "build_target_cluster_map.py"


def _run(
    tmp_path: Path,
    *extra_args: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--fasta",
            str(tmp_path / "proteins.fasta"),
            "--cluster30-tsv",
            str(tmp_path / "c30.tsv"),
            "--cluster50-tsv",
            str(tmp_path / "c50.tsv"),
            "--out-csv",
            str(tmp_path / "clusters.csv"),
            "--out-manifest",
            str(tmp_path / "clusters.manifest.json"),
            "--mmseqs-version",
            "18.8cc5c",
            *extra_args,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _seq(letter: str, length: int = 30) -> str:
    return letter * length


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_build_target_cluster_map_is_complete_and_hashed(tmp_path: Path) -> None:
    (tmp_path / "proteins.fasta").write_text(
        f">P1\n{_seq('A')}\n>P2\n{_seq('C')}\n>P3\n{_seq('G')}\n"
    )
    (tmp_path / "c30.tsv").write_text("P1\tP1\nP1\tP2\nP3\tP3\n")
    (tmp_path / "c50.tsv").write_text("P1\tP1\nP2\tP2\nP3\tP3\n")

    result = _run(tmp_path)

    assert result.returncode == 0, result.stderr
    frame = pd.read_csv(tmp_path / "clusters.csv")
    assert frame.to_dict("records") == [
        {"uniprot": "P1", "target_cluster_30": "P1", "target_cluster_50": "P1"},
        {"uniprot": "P2", "target_cluster_30": "P1", "target_cluster_50": "P2"},
        {"uniprot": "P3", "target_cluster_30": "P3", "target_cluster_50": "P3"},
    ]
    manifest = json.loads((tmp_path / "clusters.manifest.json").read_text())
    assert manifest["schema_version"] == "skinscout.target-cluster-map.v2"
    assert manifest["artifact"]["rows"] == 3
    assert manifest["artifact"]["cluster30_count"] == 2
    assert manifest["artifact"]["cluster50_count"] == 3
    assert manifest["artifact"]["sha256"]
    assert manifest["inputs"]["fasta_sha256"] == _sha256(tmp_path / "proteins.fasta")
    assert manifest["inputs"]["fasta_files"] == [
        {
            "role": "base",
            "path": str((tmp_path / "proteins.fasta").resolve()),
            "sha256": _sha256(tmp_path / "proteins.fasta"),
            "accepted_count": 3,
            "rejected_count": 0,
        }
    ]
    assert manifest["parameters"]["sequence_identity_thresholds"] == {
        "main": 0.3,
        "sensitivity": 0.5,
    }
    assert manifest["parameters"]["min_sequence_length"] == 30


def test_source_manifest_is_bound_to_the_exact_fasta(tmp_path: Path) -> None:
    fasta = tmp_path / "proteins.fasta"
    fasta.write_text(f">P1\n{_seq('A')}\n>P2\n{_seq('C')}\n")
    (tmp_path / "c30.tsv").write_text("P1\tP1\nP2\tP2\n")
    (tmp_path / "c50.tsv").write_text("P1\tP1\nP2\tP2\n")
    source_manifest = tmp_path / "source.json"
    source_manifest.write_text(
        json.dumps(
            {
                "schema_version": "skinscout.protein-sequence-source.v1",
                "source": {
                    "name": "fixture",
                    "version": "v1",
                    "license": "test",
                },
                "sequence_artifact": {
                    "path": str(fasta.resolve()),
                    "sha256": _sha256(fasta),
                    "accession_count": 2,
                },
            }
        )
        + "\n"
    )

    result = _run(tmp_path, "--source-manifest", str(source_manifest))

    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / "clusters.manifest.json").read_text())
    source_record = manifest["inputs"]["source_manifest"]
    assert source_record["sha256"] == _sha256(source_manifest)
    assert source_record["source"]["name"] == "fixture"
    assert source_record["sequence_artifact"]["accession_count"] == 2

    source = json.loads(source_manifest.read_text())
    source["sequence_artifact"]["sha256"] = "0" * 64
    source_manifest.write_text(json.dumps(source) + "\n")
    failed = _run(tmp_path, "--source-manifest", str(source_manifest))
    assert failed.returncode != 0
    assert "FASTA sha256 mismatch" in failed.stderr
    assert not (tmp_path / "clusters.csv").exists()


def test_build_target_cluster_map_rejects_missing_assignment_and_cleans_stale(
    tmp_path: Path,
) -> None:
    (tmp_path / "proteins.fasta").write_text(f">P1\n{_seq('A')}\n>P2\n{_seq('C')}\n")
    (tmp_path / "c30.tsv").write_text("P1\tP1\n")
    (tmp_path / "c50.tsv").write_text("P1\tP1\nP2\tP2\n")
    (tmp_path / "clusters.csv").write_text("stale\n")
    (tmp_path / "clusters.manifest.json").write_text("stale\n")

    result = _run(tmp_path)

    assert result.returncode != 0
    assert "is incomplete" in result.stderr
    assert not (tmp_path / "clusters.csv").exists()
    assert not (tmp_path / "clusters.manifest.json").exists()


def test_build_target_cluster_map_rejects_duplicate_member(tmp_path: Path) -> None:
    (tmp_path / "proteins.fasta").write_text(f">P1\n{_seq('A')}\n>P2\n{_seq('C')}\n")
    (tmp_path / "c30.tsv").write_text("P1\tP1\nP1\tP2\nP2\tP2\n")
    (tmp_path / "c50.tsv").write_text("P1\tP1\nP2\tP2\n")

    result = _run(tmp_path)

    assert result.returncode != 0
    assert "assigns member 'P2' more than once" in result.stderr
    assert not (tmp_path / "clusters.csv").exists()


def test_build_target_cluster_map_merges_additional_fasta(tmp_path: Path) -> None:
    (tmp_path / "proteins.fasta").write_text(f">P1\n{_seq('a')}\n")
    (tmp_path / "extra1.fasta").write_text(f">P2\n{_seq('c')}\n")
    (tmp_path / "extra2.fasta").write_text(f">P3\n{_seq('g')}\n")
    (tmp_path / "c30.tsv").write_text("P1\tP1\nP1\tP2\nP3\tP3\n")
    (tmp_path / "c50.tsv").write_text("P1\tP1\nP2\tP2\nP3\tP3\n")

    result = _run(
        tmp_path,
        "--additional-fasta",
        str(tmp_path / "extra1.fasta"),
        "--additional-fasta",
        str(tmp_path / "extra2.fasta"),
    )

    assert result.returncode == 0, result.stderr
    frame = pd.read_csv(tmp_path / "clusters.csv")
    assert frame["uniprot"].tolist() == ["P1", "P2", "P3"]
    manifest = json.loads((tmp_path / "clusters.manifest.json").read_text())
    assert manifest["inputs"]["fasta_files"] == [
        {
            "role": "base",
            "path": str((tmp_path / "proteins.fasta").resolve()),
            "sha256": _sha256(tmp_path / "proteins.fasta"),
            "accepted_count": 1,
            "rejected_count": 0,
        },
        {
            "role": "additional",
            "path": str((tmp_path / "extra1.fasta").resolve()),
            "sha256": _sha256(tmp_path / "extra1.fasta"),
            "accepted_count": 1,
            "rejected_count": 0,
        },
        {
            "role": "additional",
            "path": str((tmp_path / "extra2.fasta").resolve()),
            "sha256": _sha256(tmp_path / "extra2.fasta"),
            "accepted_count": 1,
            "rejected_count": 0,
        },
    ]


def test_build_target_cluster_map_dedupes_same_sequence_accession(
    tmp_path: Path,
) -> None:
    (tmp_path / "proteins.fasta").write_text(f">P1\n{_seq('a')}\n")
    (tmp_path / "extra.fasta").write_text(f">P1\n{_seq('A')}\n>P2\n{_seq('c')}\n")
    (tmp_path / "c30.tsv").write_text("P1\tP1\nP1\tP2\n")
    (tmp_path / "c50.tsv").write_text("P1\tP1\nP2\tP2\n")

    result = _run(tmp_path, "--additional-fasta", str(tmp_path / "extra.fasta"))

    assert result.returncode == 0, result.stderr
    frame = pd.read_csv(tmp_path / "clusters.csv")
    assert frame["uniprot"].tolist() == ["P1", "P2"]
    manifest = json.loads((tmp_path / "clusters.manifest.json").read_text())
    assert manifest["inputs"]["fasta_files"][1]["accepted_count"] == 2
    assert manifest["artifact"]["rows"] == 2


def test_build_target_cluster_map_rejects_conflicting_sequence(
    tmp_path: Path,
) -> None:
    (tmp_path / "proteins.fasta").write_text(f">P1\n{_seq('A')}\n")
    (tmp_path / "extra.fasta").write_text(f">P1\n{_seq('C')}\n")
    (tmp_path / "c30.tsv").write_text("P1\tP1\n")
    (tmp_path / "c50.tsv").write_text("P1\tP1\n")

    result = _run(tmp_path, "--additional-fasta", str(tmp_path / "extra.fasta"))

    assert result.returncode != 0
    assert "Conflicting FASTA sequence for accession 'P1'" in result.stderr
    assert not (tmp_path / "clusters.csv").exists()


def test_build_target_cluster_map_rejects_and_counts_short_sequences(
    tmp_path: Path,
) -> None:
    (tmp_path / "proteins.fasta").write_text(f">P1\n{_seq('A')}\n>P2\nSHORT\n")
    (tmp_path / "extra.fasta").write_text(f">P3\n{_seq('C')}\n>P4\nTOOSHORT\n")
    (tmp_path / "c30.tsv").write_text("P1\tP1\nP3\tP3\n")
    (tmp_path / "c50.tsv").write_text("P1\tP1\nP3\tP3\n")

    result = _run(tmp_path, "--additional-fasta", str(tmp_path / "extra.fasta"))

    assert result.returncode == 0, result.stderr
    frame = pd.read_csv(tmp_path / "clusters.csv")
    assert frame["uniprot"].tolist() == ["P1", "P3"]
    manifest = json.loads((tmp_path / "clusters.manifest.json").read_text())
    assert manifest["inputs"]["fasta_files"][0]["accepted_count"] == 1
    assert manifest["inputs"]["fasta_files"][0]["rejected_count"] == 1
    assert manifest["inputs"]["fasta_files"][1]["accepted_count"] == 1
    assert manifest["inputs"]["fasta_files"][1]["rejected_count"] == 1


def test_build_target_cluster_map_rejects_cluster_assignment_for_short_sequence(
    tmp_path: Path,
) -> None:
    (tmp_path / "proteins.fasta").write_text(f">P1\n{_seq('A')}\n>P2\nSHORT\n")
    (tmp_path / "c30.tsv").write_text("P1\tP1\nP2\tP2\n")
    (tmp_path / "c50.tsv").write_text("P1\tP1\n")

    result = _run(tmp_path)

    assert result.returncode != 0
    assert "identifier(s) absent from FASTA" in result.stderr
    assert "P2" in result.stderr


def _sequence_universe_manifest(
    tmp_path: Path,
    *,
    supplemental: Path,
    accepted: list[str],
    exclusions: list[dict[str, object]],
) -> Path:
    path = tmp_path / "sequence-universe.manifest.json"
    payload = {
        "schema_version": "skinscout.evidence-target-sequence-universe.v2",
        "inputs": {
            "uniprot_metadata": {
                "source": {
                    "name": "UniProtKB",
                    "release": "2026_02",
                    "license": "CC BY 4.0",
                }
            }
        },
        "evidence_targets": {
            "total_count": len(accepted) + len(exclusions),
        },
        "artifacts": {
            "target_fasta": {
                "path": str(supplemental.resolve()),
                "sha256": _sha256(supplemental),
                "accepted_count": len(accepted),
                "accepted_accessions": accepted,
            }
        },
        "exclusions": {"count": len(exclusions), "records": exclusions},
    }
    path.write_text(json.dumps(payload) + "\n")
    return path


def test_binds_sequence_universe_and_carries_audited_exclusions(tmp_path: Path) -> None:
    target_fasta = tmp_path / "proteins.fasta"
    target_fasta.write_text(f">P1\n{_seq('A')}\n>P2\n{_seq('C')}\n")
    (tmp_path / "c30.tsv").write_text("P1\tP1\nP1\tP2\n")
    (tmp_path / "c50.tsv").write_text("P1\tP1\nP2\tP2\n")
    exclusions = [
        {
            "uniprot": "P3",
            "reason": "unresolved_or_obsolete_uniprot_record",
            "uniprot_release": "2026_02",
        }
    ]
    sequence_manifest = _sequence_universe_manifest(
        tmp_path,
        supplemental=target_fasta,
        accepted=["P1", "P2"],
        exclusions=exclusions,
    )

    result = _run(
        tmp_path,
        "--sequence-universe-manifest",
        str(sequence_manifest),
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / "clusters.manifest.json").read_text())
    assert manifest["inputs"]["sequence_universe_manifest"]["sha256"] == _sha256(
        sequence_manifest
    )
    assert manifest["inputs"]["sequence_universe_manifest"]["accepted_count"] == 2
    assert manifest["excluded_targets"] == {
        "count": 1,
        "records": exclusions,
        "policy": (
            "No synthetic sequence cluster is assigned; only exclusions validated by the "
            "evidence target sequence universe manifest may be omitted downstream."
        ),
    }


def test_sequence_universe_fasta_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    target_fasta = tmp_path / "proteins.fasta"
    target_fasta.write_text(f">P1\n{_seq('A')}\n>P2\n{_seq('C')}\n")
    (tmp_path / "c30.tsv").write_text("P1\tP1\nP2\tP2\n")
    (tmp_path / "c50.tsv").write_text("P1\tP1\nP2\tP2\n")
    manifest_path = _sequence_universe_manifest(
        tmp_path,
        supplemental=target_fasta,
        accepted=["P1", "P2"],
        exclusions=[],
    )
    payload = json.loads(manifest_path.read_text())
    payload["artifacts"]["target_fasta"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload) + "\n")

    result = _run(
        tmp_path,
        "--sequence-universe-manifest",
        str(manifest_path),
    )

    assert result.returncode != 0
    assert "Target sequence universe FASTA sha256 mismatch" in result.stderr
    assert not (tmp_path / "clusters.csv").exists()
