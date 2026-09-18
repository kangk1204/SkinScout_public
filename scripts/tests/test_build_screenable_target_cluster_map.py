"""Tests for the evaluation-panel-independent screenable target universe."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "eval" / "build_screenable_target_cluster_map.py"
ALPHAFOLD_SOURCE = {
    "name": "AlphaFold Protein Structure Database human proteome",
    "version": "v4",
    "proteome_id": "UP000005640",
    "taxon_id": "9606",
    "archive_url": (
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v4/"
        "UP000005640_9606_HUMAN_v4.tar"
    ),
    "license": "CC BY 4.0",
    "license_url": "https://alphafold.ebi.ac.uk/faq",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, ids: list[str]) -> None:
    pd.DataFrame(
        {
            "uniprot": ids,
            "target_cluster_30": ids,
            "target_cluster_50": ids,
        }
    ).to_csv(path, index=False)


def _fixtures(tmp_path: Path) -> None:
    base_fasta = tmp_path / "base.fasta"
    base_fasta.write_text(">P1\nAAAAAAAAAA\n>P2\nCCCCCCCCCC\n")
    base_csv = tmp_path / "base.csv"
    _write_csv(base_csv, ["P1", "P2"])
    source_manifest = tmp_path / "alphafold_source.json"
    source_manifest.write_text(
        json.dumps(
            {
                "schema_version": "skinscout.protein-sequence-source.v1",
                "source": ALPHAFOLD_SOURCE,
                "sequence_artifact": {
                    "path": str(base_fasta.resolve()),
                    "sha256": _sha256(base_fasta),
                    "accession_count": 2,
                    "sequence_role": "canonical_uniprot_reference_sequence",
                },
            }
        )
        + "\n"
    )
    (tmp_path / "base.manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "skinscout.target-cluster-map.v2",
                "inputs": {
                    "fasta": str(base_fasta.resolve()),
                    "fasta_sha256": _sha256(base_fasta),
                    "source_manifest": {
                        "path": str(source_manifest.resolve()),
                        "sha256": _sha256(source_manifest),
                        "schema_version": "skinscout.protein-sequence-source.v1",
                        "source": ALPHAFOLD_SOURCE,
                        "sequence_artifact": {
                            "path": str(base_fasta.resolve()),
                            "sha256": _sha256(base_fasta),
                            "accession_count": 2,
                            "sequence_role": "canonical_uniprot_reference_sequence",
                        },
                    },
                },
                "artifact": {
                    "sha256": _sha256(base_csv),
                    "rows": 2,
                },
            }
        )
        + "\n"
    )

    evidence_fasta = tmp_path / "evidence.fasta"
    evidence_fasta.write_text(">P1\nAAAAAAAAAA\n>P3\nGGGGGGGGGG\n")
    evidence_csv = tmp_path / "evidence.csv"
    _write_csv(evidence_csv, ["P1", "P3"])
    (tmp_path / "evidence.manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "skinscout.target-cluster-map.v2",
                "inputs": {
                    "sequence_universe_manifest": {
                        "schema_version": (
                            "skinscout.evidence-target-sequence-universe.v2"
                        ),
                        "accepted_count": 2,
                        "target_fasta_path": str(evidence_fasta.resolve()),
                        "target_fasta_sha256": _sha256(evidence_fasta),
                        "uniprot_source": {
                            "name": "UniProtKB",
                            "release": "2026_02",
                            "license": "CC BY 4.0",
                        },
                    }
                },
                "artifact": {
                    "sha256": _sha256(evidence_csv),
                    "rows": 2,
                },
                "excluded_targets": {
                    "count": 1,
                    "policy": "fixture evidence-sequence exclusions",
                    "records": [
                        {
                            "uniprot": "P_OBSOLETE",
                            "reason": "unresolved_or_obsolete_uniprot_record",
                            "uniprot_release": "2026_02",
                        }
                    ],
                },
            }
        )
        + "\n"
    )
    (tmp_path / "supplemental.fasta").write_text(">P3\nGGGGGGGGGG\n")
    (tmp_path / "c30.tsv").write_text("P1\tP1\nP1\tP2\nP3\tP3\n")
    (tmp_path / "c50.tsv").write_text("P1\tP1\nP2\tP2\nP3\tP3\n")


def _run(tmp_path: Path, *, fixture_mode: bool = True) -> subprocess.CompletedProcess[str]:
    fixture_args = ["--fixture-mode"] if fixture_mode else []
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--base-target-csv",
            str(tmp_path / "base.csv"),
            "--base-target-manifest",
            str(tmp_path / "base.manifest.json"),
            "--evidence-target-csv",
            str(tmp_path / "evidence.csv"),
            "--evidence-target-manifest",
            str(tmp_path / "evidence.manifest.json"),
            "--supplemental-fasta",
            str(tmp_path / "supplemental.fasta"),
            "--cluster30-tsv",
            str(tmp_path / "c30.tsv"),
            "--cluster50-tsv",
            str(tmp_path / "c50.tsv"),
            "--mmseqs-version",
            "18.8cc5c",
            "--out-csv",
            str(tmp_path / "screenable.csv"),
            "--out-manifest",
            str(tmp_path / "screenable.manifest.json"),
            *fixture_args,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_builds_union_without_evaluation_panel_input(tmp_path: Path) -> None:
    _fixtures(tmp_path)

    result = _run(tmp_path)

    assert result.returncode == 0, result.stderr
    frame = pd.read_csv(tmp_path / "screenable.csv")
    assert frame["uniprot"].tolist() == ["P1", "P2", "P3"]
    manifest = json.loads((tmp_path / "screenable.manifest.json").read_text())
    assert manifest["schema_version"] == "skinscout.screenable-target-cluster-map.v2"
    assert manifest["universe_policy"] == {
        "candidate_definition": (
            "union of the independent AlphaFold human-proteome receptor snapshot "
            "and claim-grade activity-evidence targets absent from that snapshot"
        ),
        "evaluation_panel_used": False,
        "known_target_assistance": False,
        "base_target_count": 2,
        "evidence_target_count": 2,
        "supplemental_target_count": 1,
        "union_target_count": 3,
    }
    assert manifest["sources"]["independent_base"]["version"] == "v4"
    assert manifest["sources"]["evidence_sequence_source"]["release"] == "2026_02"
    assert manifest["artifact"]["sha256"] == _sha256(tmp_path / "screenable.csv")
    assert manifest["excluded_targets"]["count"] == 1
    assert manifest["excluded_targets"]["records"][0]["uniprot"] == "P_OBSOLETE"
    assert manifest["production_contract"] == {
        "profile": "alphafold-human-v4-plus-claim-grade-evidence",
        "fixture_mode": True,
        "expected_base_target_count": 20171,
        "expected_union_target_count": 20204,
        "passes": False,
    }


def test_rejects_supplemental_accession_not_derived_from_evidence_union(
    tmp_path: Path,
) -> None:
    _fixtures(tmp_path)
    (tmp_path / "supplemental.fasta").write_text(">P4\nTTTTTTTTTT\n")
    (tmp_path / "screenable.csv").write_text("stale\n")
    (tmp_path / "screenable.manifest.json").write_text("stale\n")

    result = _run(tmp_path)

    assert result.returncode != 0
    assert "supplemental FASTA must exactly equal" in result.stderr
    assert not (tmp_path / "screenable.csv").exists()
    assert not (tmp_path / "screenable.manifest.json").exists()


def test_rejects_missing_uniprot_release_license_provenance(tmp_path: Path) -> None:
    _fixtures(tmp_path)
    manifest_path = tmp_path / "evidence.manifest.json"
    payload = json.loads(manifest_path.read_text())
    payload["inputs"]["sequence_universe_manifest"]["uniprot_source"][
        "license"
    ] = ""
    manifest_path.write_text(json.dumps(payload) + "\n")

    result = _run(tmp_path)

    assert result.returncode != 0
    assert "release/license provenance is invalid" in result.stderr
    assert not (tmp_path / "screenable.csv").exists()


def test_production_mode_rejects_reduced_target_universe(tmp_path: Path) -> None:
    _fixtures(tmp_path)

    result = _run(tmp_path, fixture_mode=False)

    assert result.returncode != 0
    assert "requires exactly 20171 AlphaFold base targets and 20204 union targets" in (
        result.stderr
    )
    assert not (tmp_path / "screenable.csv").exists()


def test_rejects_mislabeled_or_unbound_alphafold_source(tmp_path: Path) -> None:
    _fixtures(tmp_path)
    source_path = tmp_path / "alphafold_source.json"
    source = json.loads(source_path.read_text())
    source["source"]["version"] = "v3"
    source_path.write_text(json.dumps(source) + "\n")
    base_manifest_path = tmp_path / "base.manifest.json"
    base_manifest = json.loads(base_manifest_path.read_text())
    base_manifest["inputs"]["source_manifest"]["sha256"] = _sha256(source_path)
    base_manifest["inputs"]["source_manifest"]["source"]["version"] = "v3"
    base_manifest_path.write_text(json.dumps(base_manifest) + "\n")

    result = _run(tmp_path)

    assert result.returncode != 0
    assert "must be the pinned AlphaFold human proteome v4" in result.stderr
    assert not (tmp_path / "screenable.csv").exists()


def test_rejects_exclusion_that_also_has_evidence_cluster_assignment(
    tmp_path: Path,
) -> None:
    _fixtures(tmp_path)
    manifest_path = tmp_path / "evidence.manifest.json"
    payload = json.loads(manifest_path.read_text())
    payload["excluded_targets"]["records"][0]["uniprot"] = "P3"
    manifest_path.write_text(json.dumps(payload) + "\n")

    result = _run(tmp_path)

    assert result.returncode != 0
    assert "unexpectedly has a cluster assignment: P3" in result.stderr
    assert not (tmp_path / "screenable.csv").exists()
