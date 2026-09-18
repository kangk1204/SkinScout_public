"""F13 regression tests for the shared Stage 0 required-path content contract.

Deleting or changing any required completion artifact must fail both the
initial readiness preflight and the final Stage 0 verifier. The SkinScore axes
sidecar is part of the same path contract, and the historical
``stage0_complete.flag`` touch file must never make readiness pass by itself.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from Bio.SeqUtils.CheckSum import crc64

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import data_readiness  # noqa: E402
import stage0_verify  # noqa: E402
from stage0_manifest import (  # noqa: E402
    MANIFEST_FILENAME,
    build_manifest,
    write_manifest,
)
from stage0_verify import (  # noqa: E402
    chk_canonical_sequences,
    chk_pdbqt_roundtrip,
    chk_skin_score_axes,
)

DIGEST = "0" * 64
PDBQT_POLICY = {"min_success_fraction": 0.8, "min_success_count": 1}
AXES_SIDECAR: dict[str, object] = {
    "schema_version": "skinscout.skin-score-axes.v2",
    "score_semantics": "relative_within_build_expression_context",
    "normalization": "per-axis min-max across proteins in this build",
    "declared_weights": {
        "hpa_tissue": 0.30,
        "hpa_cell": 0.25,
        "proteome": 0.20,
        "gtex": 0.10,
        "sc": 0.15,
    },
    "effective_weights": {
        "hpa_tissue": 0.30,
        "hpa_cell": 0.25,
        "proteome": 0.20,
        "gtex": 0.10,
        "sc": 0.15,
    },
    "axes_absent": [],
    "context_support_threshold": 0.2,
}


def _config(tmp_path: Path) -> dict[str, object]:
    return {
        "paths": {
            "mmseqs": str(tmp_path / "data" / "mmseqs"),
            "uniprot_human_fasta": str(
                tmp_path / "data" / "mmseqs" / "human_canonical.fasta"
            ),
            "alphafold_clean": str(tmp_path / "data" / "human_clean"),
            "pdbqt": str(tmp_path / "data" / "human_pdbqt"),
            "docking_boxes": str(tmp_path / "data" / "docking_boxes"),
            "no_pocket_list": str(tmp_path / "data" / "no_pocket_targets.list"),
        },
        "paths_v3": {
            "skin_expression": str(tmp_path / "data" / "skin_expression"),
            "cosing": str(tmp_path / "data" / "cosing"),
            "drug_avoidance": str(tmp_path / "data" / "drug_avoidance"),
            "skin_kg": str(tmp_path / "data" / "skin_efficacy_kg"),
        },
    }


def _required_artifacts(tmp_path: Path) -> dict[str, Path]:
    config = _config(tmp_path)
    return {
        label: path
        for label, path in data_readiness.required_data_artifacts(
            "target-id",
            "comprehensive",
            config=config,
            root=tmp_path,
        )
    }


def _write_skin_score_axes(tmp_path: Path) -> tuple[Path, Path]:
    skin = tmp_path / "data" / "skin_expression"
    skin.mkdir(parents=True, exist_ok=True)
    tsv = skin / "skin_score.tsv"
    tsv.write_text("uniprot\tskin_score\nP02533\t0.5\n")
    sidecar = skin / "skin_score.tsv.axes.json"
    sidecar.write_text(json.dumps(AXES_SIDECAR))
    return tsv, sidecar


def _write_canonical_artifact(
    root: Path,
    sequences: dict[str, str],
) -> tuple[Path, Path, Path]:
    mmseqs = root / "data" / "mmseqs"
    clean_dir = root / "data" / "human_clean"
    mmseqs.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    fasta = mmseqs / "human_canonical.fasta"
    fasta_bytes = "".join(
        f">{accession}\n{sequence}\n"
        for accession, sequence in sorted(sequences.items())
    ).encode("ascii")
    fasta.write_bytes(fasta_bytes)
    for accession in sequences:
        (clean_dir / f"{accession}_clean.pdb").write_text("ATOM\n")
    records = []
    for index, (accession, sequence) in enumerate(sorted(sequences.items()), start=1):
        records.append(
            {
                "accession": accession,
                "length": len(sequence),
                "sequence_sha256": hashlib.sha256(
                    sequence.encode("ascii")
                ).hexdigest(),
                "uniprot_crc64": crc64(sequence).removeprefix("CRC-"),
                "source_fragments": [
                    {
                        "path": f"AF-{accession}-F1-model_v4.cif.gz",
                        "compressed_sha256": f"{index:064x}",
                        "seq_db_align_begin": 1,
                        "seq_db_align_end": len(sequence),
                    }
                ],
            }
        )
    manifest = {
        "schema_version": "skinscout.afdb-v4-canonical-sequences.v1",
        "provenance": {
            "source": (
                "AlphaFold Protein Structure Database human proteome v4 "
                "raw mmCIF files"
            ),
            "derivation": (
                "Canonical sequences assembled offline from AFDB fragment "
                "metadata and validated against the full UniProt CRC64 embedded "
                "by AFDB; this is not an independently downloaded UniProt FASTA "
                "snapshot."
            ),
            "alphafold_directory": str(
                (root / "data" / "alphafold_human_v4").resolve()
            ),
            "source_pattern": "*.cif.gz",
        },
        "validation": {
            "taxonomy_id": "9606",
            "checksum_algorithm": "Bio.SeqUtils.CheckSum.crc64",
            "fragment_policy": (
                "all fragments; exact spans; equal overlaps; gapless from residue 1"
            ),
            "unknown_residues_allowed": False,
        },
        "counts": {
            "source_files": len(sequences),
            "canonical_sequences": len(sequences),
            "required_receptors": len(sequences),
            "required_receptors_covered": len(sequences),
        },
        "required_receptor_coverage": {
            "directory": str(clean_dir.resolve()),
            "missing_accessions": [],
        },
        "artifact": {
            "path": str(fasta.resolve()),
            "sha256": hashlib.sha256(fasta_bytes).hexdigest(),
            "bytes": len(fasta_bytes),
            "sequence_count": len(sequences),
        },
        "sequences": records,
    }
    manifest_path = mmseqs / "human_canonical.fasta.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return fasta, manifest_path, clean_dir


def _write_receptor_manifest(root: Path) -> Path:
    pdbqt = root / "data" / "human_pdbqt"
    pdbqt.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "uniprot": "P00000",
            "status": "ok",
            "method": "meeko",
            "reason": None,
            "clean_pdb_sha256": DIGEST,
            "pocket_json_sha256": DIGEST,
        }
    ]
    (pdbqt / "P00000.pdbqt").write_text("REMARK ok\n")
    manifest = pdbqt / MANIFEST_FILENAME
    write_manifest(
        manifest,
        build_manifest(
            scope="full",
            policy=PDBQT_POLICY,
            records=records,
        ),
    )
    return manifest


def test_axes_sidecar_is_part_of_the_required_path_contract(tmp_path: Path) -> None:
    artifacts = _required_artifacts(tmp_path)

    assert "skin-expression score axes sidecar" in artifacts
    assert artifacts["skin-expression score axes sidecar"] == (
        tmp_path / "data" / "skin_expression" / "skin_score.tsv.axes.json"
    )


def test_missing_axes_sidecar_fails_initial_readiness_and_verifier(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    tsv, sidecar = _write_skin_score_axes(tmp_path)
    sidecar.unlink()

    payload = data_readiness.data_readiness_payload(
        "target-id",
        "comprehensive",
        True,
        config=config,
        root=tmp_path,
    )
    assert payload["status"] == "failed"
    assert any(
        check["label"] == "skin-expression score axes sidecar"
        and check["status"] == "missing"
        for check in payload["checks"]
    )
    assert not chk_skin_score_axes(tsv, "v3_skin_score_axes").ok


def test_changed_axes_sidecar_fails_initial_readiness_and_verifier(
    tmp_path: Path,
) -> None:
    readiness_label = "skin-expression score axes sidecar"
    tsv, sidecar = _write_skin_score_axes(tmp_path)
    assert data_readiness.check_required_artifact(
        readiness_label, sidecar
    ).status == "present"
    assert chk_skin_score_axes(tsv, "v3_skin_score_axes").ok

    payload = json.loads(sidecar.read_text())
    payload["schema_version"] = "skinscout.skin-score-axes.v1"
    sidecar.write_text(json.dumps(payload))

    readiness = data_readiness.check_required_artifact(readiness_label, sidecar)
    verifier = chk_skin_score_axes(tsv, "v3_skin_score_axes")
    assert readiness.status == "invalid"
    assert not verifier.ok
    assert "schema_version" in readiness.detail
    assert "schema_version" in verifier.detail


def test_missing_canonical_fasta_pair_fails_initial_readiness_and_verifier(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    fasta, manifest, clean_dir = _write_canonical_artifact(
        tmp_path,
        {"P12345": "ACDEFG"},
    )
    assert data_readiness.check_required_artifact(
        "canonical human FASTA", fasta
    ).status == "present"
    assert data_readiness.check_required_artifact(
        "canonical human FASTA manifest", manifest
    ).status == "present"

    fasta.unlink()
    manifest.unlink()

    payload = data_readiness.data_readiness_payload(
        "target-id",
        "comprehensive",
        True,
        config=config,
        root=tmp_path,
    )
    failed_labels = {
        check["label"] for check in payload["checks"] if check["status"] != "present"
    }
    assert {
        "canonical human FASTA",
        "canonical human FASTA manifest",
    } <= failed_labels
    assert not chk_canonical_sequences(fasta, manifest, clean_dir).ok


def test_changed_canonical_fasta_fails_initial_readiness_and_verifier(
    tmp_path: Path,
) -> None:
    fasta, manifest, clean_dir = _write_canonical_artifact(
        tmp_path,
        {"P12345": "ACDEFG"},
    )
    fasta.write_text(">P12345\nACDEFA\n")

    for label, artifact in (
        ("canonical human FASTA", fasta),
        ("canonical human FASTA manifest", manifest),
    ):
        assert data_readiness.check_required_artifact(label, artifact).status == (
            "invalid"
        )
    assert not chk_canonical_sequences(fasta, manifest, clean_dir).ok


def test_changed_canonical_manifest_fails_initial_readiness_and_verifier(
    tmp_path: Path,
) -> None:
    fasta, manifest, clean_dir = _write_canonical_artifact(
        tmp_path,
        {"P12345": "ACDEFG"},
    )
    payload = json.loads(manifest.read_text())
    payload["artifact"]["sha256"] = DIGEST
    manifest.write_text(json.dumps(payload))

    for label, artifact in (
        ("canonical human FASTA", fasta),
        ("canonical human FASTA manifest", manifest),
    ):
        assert data_readiness.check_required_artifact(label, artifact).status == (
            "invalid"
        )
    assert not chk_canonical_sequences(fasta, manifest, clean_dir).ok


def test_manifest_coverage_directory_mismatch_fails_initial_readiness(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _fasta, manifest, _clean_dir = _write_canonical_artifact(
        tmp_path,
        {"P12345": "ACDEFG"},
    )
    payload = json.loads(manifest.read_text())
    payload["required_receptor_coverage"]["directory"] = str(
        tmp_path / "somewhere_else"
    )
    manifest.write_text(json.dumps(payload))

    readiness = data_readiness.check_required_artifact(
        "canonical human FASTA manifest",
        manifest,
        clean_dir=tmp_path / "data" / "human_clean",
    )
    assert readiness.status == "invalid"

    failed = data_readiness.data_readiness_payload(
        "target-id",
        "comprehensive",
        True,
        config=config,
        root=tmp_path,
    )
    assert failed["status"] == "failed"
    assert any(
        check["label"] == "canonical human FASTA manifest"
        and check["status"] == "invalid"
        for check in failed["checks"]
    )


def test_missing_receptor_prep_manifest_fails_initial_readiness_and_verifier(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    manifest = _write_receptor_manifest(tmp_path)
    pdbqt = manifest.parent
    assert data_readiness.check_required_artifact(
        "PDBQT receptor marker", manifest
    ).status == "present"
    assert chk_pdbqt_roundtrip(
        pdbqt,
        manifest_path=manifest,
        expected_policy=PDBQT_POLICY,
    ).ok

    manifest.unlink()

    payload = data_readiness.data_readiness_payload(
        "target-id",
        "comprehensive",
        True,
        config=config,
        root=tmp_path,
    )
    assert any(
        check["label"] == "PDBQT receptor marker" and check["status"] == "missing"
        for check in payload["checks"]
    )
    assert not chk_pdbqt_roundtrip(
        pdbqt,
        manifest_path=manifest,
        expected_policy=PDBQT_POLICY,
    ).ok


def test_tampered_receptor_prep_manifest_fails_initial_readiness_and_verifier(
    tmp_path: Path,
) -> None:
    manifest = _write_receptor_manifest(tmp_path)
    pdbqt = manifest.parent
    payload = json.loads(manifest.read_text())
    payload["counts"]["success"] = 0
    manifest.write_text(json.dumps(payload))

    readiness = data_readiness.check_required_artifact(
        "PDBQT receptor marker", manifest
    )
    assert readiness.status == "invalid"
    assert not chk_pdbqt_roundtrip(
        pdbqt,
        manifest_path=manifest,
        expected_policy=PDBQT_POLICY,
    ).ok


def test_stale_completion_flag_alone_never_passes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    manifests = tmp_path / "data" / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    flag = manifests / "stage0_complete.flag"
    flag.touch()

    assert stage0_verify.chk_flag(manifests).ok

    payload = data_readiness.data_readiness_payload(
        "target-id",
        "comprehensive",
        True,
        config=config,
        root=tmp_path,
    )
    assert payload["status"] == "failed"
    assert any(
        check["label"].startswith("canonical human FASTA")
        for check in payload["checks"]
    )

    labels = set(_required_artifacts(tmp_path))
    assert not any("flag" in label.lower() for label in labels)

    assert stage0_verify.run_all(
        tmp_path,
        strict=True,
        claim_quality=True,
        paths=data_readiness.resolve_stage0_paths(config, root=tmp_path),
        min_cosing_rows=1,
        min_drug_rows=1,
        min_skin_score_rows=1,
        min_kg_gene_edges=1,
    ) == 1
