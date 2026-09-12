"""Regression tests for claim-quality Stage 0 reference gates."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest
from Bio.SeqUtils.CheckSum import crc64

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage0_verify import (  # noqa: E402
    chk_canonical_sequences,
    chk_p2rank,
    chk_pdbqt_roundtrip,
    chk_kg_claim_quality,
    chk_mmseqs_training_cutoff,
    chk_table_claim_quality,
    collect_checks,
    run_all,
)


def _write_canonical_artifact(
    root: Path,
    sequences: dict[str, str],
) -> tuple[Path, Path, Path]:
    mmseqs = root / "data" / "mmseqs"
    clean_dir = root / "data" / "human_clean"
    mmseqs.mkdir(parents=True)
    clean_dir.mkdir(parents=True)
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
                "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
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
            "source": "AlphaFold Protein Structure Database human proteome v4 raw mmCIF files",
            "derivation": (
                "Canonical sequences assembled offline from AFDB fragment metadata and validated "
                "against the full UniProt CRC64 embedded by AFDB; this is not an independently "
                "downloaded UniProt FASTA snapshot."
            ),
            "alphafold_directory": str((root / "data" / "alphafold_human_v4").resolve()),
            "source_pattern": "*.cif.gz",
        },
        "validation": {
            "taxonomy_id": "9606",
            "checksum_algorithm": "Bio.SeqUtils.CheckSum.crc64",
            "fragment_policy": "all fragments; exact spans; equal overlaps; gapless from residue 1",
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


def test_canonical_sequences_accepts_builder_manifest_contract(tmp_path: Path) -> None:
    fasta, manifest, clean_dir = _write_canonical_artifact(
        tmp_path,
        {"P12345": "ACDEFG", "Q99999": "MNPQRS"},
    )

    check = chk_canonical_sequences(fasta, manifest, clean_dir)

    assert check.ok, check.detail
    assert "sequences=2" in check.detail


@pytest.mark.parametrize(
    ("field_path", "bad_value", "expected_detail"),
    [
        (("artifact", "sha256"), "0" * 64, "artifact.sha256"),
        (("artifact", "bytes"), 1, "artifact.bytes"),
        (("artifact", "sequence_count"), 99, "artifact.sequence_count"),
        (("counts", "canonical_sequences"), 99, "counts.canonical_sequences"),
        (("sequences", 0, "sequence_sha256"), "0" * 64, "sequence_sha256:P12345"),
        (("sequences", 0, "uniprot_crc64"), "0" * 16, "uniprot_crc64:P12345"),
        (("provenance", "source"), "UniProt download", "provenance.source"),
    ],
)
def test_canonical_sequences_rejects_tampered_manifest_fields(
    tmp_path: Path,
    field_path: tuple[object, ...],
    bad_value: object,
    expected_detail: str,
) -> None:
    fasta, manifest_path, clean_dir = _write_canonical_artifact(
        tmp_path,
        {"P12345": "ACDEFG"},
    )
    manifest = json.loads(manifest_path.read_text())
    target = manifest
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = bad_value
    manifest_path.write_text(json.dumps(manifest))

    check = chk_canonical_sequences(fasta, manifest_path, clean_dir)

    assert not check.ok
    assert expected_detail in check.detail


def test_canonical_sequences_rejects_stale_or_tampered_fasta(tmp_path: Path) -> None:
    fasta, manifest, clean_dir = _write_canonical_artifact(
        tmp_path,
        {"P12345": "ACDEFG"},
    )
    fasta.write_text(">P12345\nACDEFA\n")

    check = chk_canonical_sequences(fasta, manifest, clean_dir)

    assert not check.ok
    assert "artifact.sha256" in check.detail
    assert "sequence_sha256:P12345" in check.detail
    assert "uniprot_crc64:P12345" in check.detail


def test_canonical_sequences_requires_all_cleaned_receptors(tmp_path: Path) -> None:
    fasta, manifest, clean_dir = _write_canonical_artifact(
        tmp_path,
        {"P12345": "ACDEFG"},
    )
    (clean_dir / "Q99999_clean.pdb").write_text("ATOM\n")

    check = chk_canonical_sequences(fasta, manifest, clean_dir)

    assert not check.ok
    assert "required_receptors.missing:Q99999" in check.detail
    assert "counts.required_receptors" in check.detail


def test_collect_checks_claim_quality_includes_canonical_sequence_gate(tmp_path: Path) -> None:
    checks = collect_checks(tmp_path, claim_quality=True, check_stage0_flag=False)

    by_name = {check.name: check for check in checks}
    assert "claim_canonical_human_sequences" in by_name
    assert not by_name["claim_canonical_human_sequences"].ok


def test_claim_quality_cosing_rejects_placeholder_stub(tmp_path: Path) -> None:
    cosing = tmp_path / "cosing.parquet"
    pd.DataFrame([
        {
            "inci_name": "Niacinamide",
            "smiles": "NC(=O)c1cccnc1",
            "inchikey": "DFPAKSUCGFBDDF-UHFFFAOYSA-N",
            "ecfp4": [0] * 32,
            "scaffold_smiles": "c1ccncc1",
        }
    ]).to_parquet(cosing)

    check = chk_table_claim_quality(
        cosing,
        "claim_cosing_reference",
        {"inci_name", "smiles", "inchikey", "ecfp4", "scaffold_smiles"},
        min_rows=100,
        placeholder_markers={"inci_name": {"Niacinamide", "Retinol", "Glycerin"}},
    )

    assert not check.ok
    assert "rows=1" in check.detail
    assert "inci_name=Niacinamide" in check.detail


def test_claim_quality_reference_passes_when_non_placeholder_and_large_enough(tmp_path: Path) -> None:
    ref = tmp_path / "reference.parquet"
    pd.DataFrame([
        {
            "drug_id": f"DRUG{i}",
            "name": f"Drug {i}",
            "smiles": "CCO",
            "inchikey": f"KEY{i}",
            "ecfp4": [0] * 32,
        }
        for i in range(5)
    ]).to_parquet(ref)

    check = chk_table_claim_quality(
        ref,
        "claim_drug_reference",
        {"drug_id", "name", "smiles", "inchikey", "ecfp4"},
        min_rows=5,
        placeholder_markers={"drug_id": {"PLACEHOLDER", "PLACEHOLDER1"}},
    )

    assert check.ok, check.detail


def test_claim_quality_requires_mmseqs_training_cutoff_db(tmp_path: Path) -> None:
    mmseqs = tmp_path / "mmseqs"
    mmseqs.mkdir()
    (mmseqs / "human_db.dbtype").write_text("")
    (mmseqs / "human_db.lookup").write_text("")

    check = chk_mmseqs_training_cutoff(mmseqs)

    assert not check.ok
    assert "training_cutoff_seqs.fasta" in check.detail
    assert "training_cutoff_db.dbtype" in check.detail


def test_claim_quality_requires_no_pocket_target_artifact(tmp_path: Path) -> None:
    checks = collect_checks(
        tmp_path,
        claim_quality=True,
        check_stage0_flag=False,
        min_cosing_rows=1,
        min_drug_rows=1,
        min_skin_score_rows=1,
        min_kg_gene_edges=1,
    )
    by_name = {check.name: check for check in checks}

    assert not by_name["claim_no_pocket_targets_list"].ok
    assert "missing" in by_name["claim_no_pocket_targets_list"].detail


def test_claim_quality_mmseqs_training_cutoff_db_passes_when_complete(tmp_path: Path) -> None:
    mmseqs = tmp_path / "mmseqs"
    mmseqs.mkdir()
    for name in (
        "training_cutoff_seqs.fasta",
        "training_cutoff_db.dbtype",
        "training_cutoff_db.lookup",
    ):
        (mmseqs / name).write_text("ok\n")

    check = chk_mmseqs_training_cutoff(mmseqs)

    assert check.ok, check.detail


def _write_pocket_manifest(path: Path, target: str, *, pockets: bool = True) -> None:
    if pockets:
        path.write_text(
            f'{{"uniprot": "{target}", "pockets": ['
            '{"rank": 1, "score": 2.0, "druggability": 0.8, '
            '"center": [1.0, 2.0, 3.0], "radius": 12.0}]}'
            "\n"
        )
    else:
        path.write_text(f'{{"uniprot": "{target}", "pockets": []}}\n')


def test_p2rank_rejects_empty_directory(tmp_path: Path) -> None:
    pocket_dir = tmp_path / "human_pockets"
    pocket_dir.mkdir()

    check = chk_p2rank(pocket_dir)

    assert not check.ok
    assert "no pocket manifests found" in check.detail


def test_p2rank_checks_after_legacy_sample_boundary(tmp_path: Path) -> None:
    pocket_dir = tmp_path / "human_pockets"
    pocket_dir.mkdir()
    for idx in range(501):
        _write_pocket_manifest(pocket_dir / f"P{idx:05d}.pockets.json", f"P{idx:05d}")
    (pocket_dir / "P00500.pockets.json").write_text("{not-json\n")

    check = chk_p2rank(pocket_dir)

    assert not check.ok
    assert "501 pocket manifests" in check.detail
    assert "invalid/schema issue" in check.detail


def test_p2rank_rejects_target_universe_mismatch(tmp_path: Path) -> None:
    pocket_dir = tmp_path / "human_pockets"
    pocket_dir.mkdir()
    _write_pocket_manifest(pocket_dir / "P00001.pockets.json", "P00001")
    _write_pocket_manifest(pocket_dir / "P99999.pockets.json", "P99999")

    check = chk_p2rank(
        pocket_dir,
        expected_targets={"P00001", "P00002"},
    )

    assert not check.ok
    assert "missing=1" in check.detail
    assert "unexpected=1" in check.detail
    assert "P00002" in check.detail
    assert "P99999" in check.detail


def test_p2rank_rejects_manifest_identity_mismatch(tmp_path: Path) -> None:
    pocket_dir = tmp_path / "human_pockets"
    pocket_dir.mkdir()
    _write_pocket_manifest(
        pocket_dir / "P00001.pockets.json",
        "P99999",
    )

    check = chk_p2rank(pocket_dir)

    assert not check.ok
    assert "1 invalid/schema issue" in check.detail


def test_pdbqt_roundtrip_rejects_empty_directory(tmp_path: Path) -> None:
    pdbqt_dir = tmp_path / "human_pdbqt"
    pdbqt_dir.mkdir()

    check = chk_pdbqt_roundtrip(pdbqt_dir)

    assert not check.ok
    assert "no PDBQT files found" in check.detail


def test_pdbqt_roundtrip_checks_after_legacy_sample_boundary(tmp_path: Path) -> None:
    pdbqt_dir = tmp_path / "human_pdbqt"
    pdbqt_dir.mkdir()
    for idx in range(201):
        (pdbqt_dir / f"P{idx:05d}.pdbqt").write_text("REMARK ok\n")
    (pdbqt_dir / "P00200.pdbqt").write_text("BROKEN\n")

    check = chk_pdbqt_roundtrip(pdbqt_dir)

    assert not check.ok
    assert "201 pdbqt" in check.detail
    assert "suspect headers" in check.detail


def test_pdbqt_roundtrip_rejects_with_pocket_target_mismatch(tmp_path: Path) -> None:
    pdbqt_dir = tmp_path / "human_pdbqt"
    pdbqt_dir.mkdir()
    (pdbqt_dir / "P00001.pdbqt").write_text("REMARK ok\n")
    (pdbqt_dir / "P99999.pdbqt").write_text("REMARK ok\n")

    check = chk_pdbqt_roundtrip(
        pdbqt_dir,
        expected_targets={"P00001", "P00002"},
    )

    assert not check.ok
    assert "missing=1" in check.detail
    assert "unexpected=1" in check.detail
    assert "P00002" in check.detail
    assert "P99999" in check.detail


def test_collect_checks_claim_quality_validates_pocket_and_pdbqt_completeness(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data = tmp_path / "data"
    clean_dir = data / "human_clean"
    pocket_dir = data / "human_pockets"
    pdbqt_dir = data / "human_pdbqt"
    clean_dir.mkdir(parents=True)
    pocket_dir.mkdir()
    pdbqt_dir.mkdir()
    for target in ("P00001", "P00002", "P00003"):
        (clean_dir / f"{target}_clean.pdb").write_text("ATOM\n" * 80)
    _write_pocket_manifest(pocket_dir / "P00001.pockets.json", "P00001")
    _write_pocket_manifest(pocket_dir / "P00002.pockets.json", "P00002")
    _write_pocket_manifest(pocket_dir / "P00003.pockets.json", "P00003", pockets=False)
    (data / "no_pocket_targets.list").write_text("P00003\n")
    (pdbqt_dir / "P00001.pdbqt").write_text("REMARK ok\n")

    monkeypatch.setattr(
        "stage0_verify.chk_alphafold_count",
        lambda path: type("CheckObj", (), {"name": "alphafold_count", "ok": True, "detail": str(path)})(),
    )
    monkeypatch.setattr(
        "stage0_verify.chk_cleaned_count",
        lambda path: type("CheckObj", (), {"name": "cleaned_count_and_size", "ok": True, "detail": str(path)})(),
    )
    monkeypatch.setattr(
        "stage0_verify.chk_mmseqs",
        lambda path: type("CheckObj", (), {"name": "mmseqs_index", "ok": True, "detail": str(path)})(),
    )
    monkeypatch.setattr(
        "stage0_verify.chk_table",
        lambda path, name, required_cols: type("CheckObj", (), {"name": name, "ok": True, "detail": str(path)})(),
    )
    monkeypatch.setattr(
        "stage0_verify.chk_graphml",
        lambda path, name: type("CheckObj", (), {"name": name, "ok": True, "detail": str(path)})(),
    )
    monkeypatch.setattr(
        "stage0_verify.chk_nonempty_file",
        lambda path, name: type("CheckObj", (), {"name": name, "ok": True, "detail": str(path)})(),
    )

    checks = collect_checks(tmp_path, claim_quality=True, check_stage0_flag=False)
    by_name = {check.name: check for check in checks}

    assert by_name["p2rank_no_nan"].ok, by_name["p2rank_no_nan"].detail
    assert not by_name["pdbqt_roundtrip"].ok
    assert "P00002" in by_name["pdbqt_roundtrip"].detail


def test_claim_quality_kg_rejects_seed_only_graph(tmp_path: Path) -> None:
    graph_path = tmp_path / "kg.graphml"
    graph = nx.MultiDiGraph()
    graph.add_node("category:hydration", type="EfficacyCategory", name="hydration")
    graph.add_node("gene:P20930", type="Gene", uniprot="P20930")
    graph.add_edge("gene:P20930", "category:hydration", relation="ASSOCIATED_WITH")
    nx.write_graphml(graph, graph_path)

    check = chk_kg_claim_quality(graph_path, min_gene_edges=1)

    assert not check.ok
    assert "pubtator_backed_nodes=0" in check.detail


def test_claim_quality_kg_passes_with_pubtator_backed_categories(tmp_path: Path) -> None:
    graph_path = tmp_path / "kg.graphml"
    graph = nx.MultiDiGraph()
    graph.add_node(
        "category:hydration",
        type="EfficacyCategory",
        name="hydration",
        pmid_count=20,
        sample_pmids="1;2;3",
    )
    for i in range(3):
        graph.add_node(f"gene:P{i}", type="Gene", uniprot=f"P{i}")
        graph.add_edge(f"gene:P{i}", "category:hydration", relation="ASSOCIATED_WITH")
    nx.write_graphml(graph, graph_path)

    check = chk_kg_claim_quality(graph_path, min_gene_edges=3)

    assert check.ok, check.detail


def test_run_all_can_skip_stage0_flag_while_creating_it(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data = tmp_path / "data"
    monkeypatch.setattr(
        "stage0_verify.chk_alphafold_count",
        lambda path: type("CheckObj", (), {"name": "alphafold_count", "ok": True, "detail": str(path)})(),
    )
    monkeypatch.setattr(
        "stage0_verify.chk_cleaned_count",
        lambda path: type("CheckObj", (), {"name": "cleaned_count_and_size", "ok": True, "detail": str(path)})(),
    )
    monkeypatch.setattr(
        "stage0_verify.chk_p2rank",
        lambda path: type("CheckObj", (), {"name": "p2rank_no_nan", "ok": True, "detail": str(path)})(),
    )
    monkeypatch.setattr(
        "stage0_verify.chk_pdbqt_roundtrip",
        lambda path: type("CheckObj", (), {"name": "pdbqt_roundtrip", "ok": True, "detail": str(path)})(),
    )
    monkeypatch.setattr(
        "stage0_verify.chk_mmseqs",
        lambda path: type("CheckObj", (), {"name": "mmseqs_index", "ok": True, "detail": str(path)})(),
    )
    monkeypatch.setattr(
        "stage0_verify.chk_table",
        lambda path, name, required_cols: type("CheckObj", (), {"name": name, "ok": True, "detail": str(path)})(),
    )
    monkeypatch.setattr(
        "stage0_verify.chk_graphml",
        lambda path, name: type("CheckObj", (), {"name": name, "ok": True, "detail": str(path)})(),
    )
    monkeypatch.setattr(
        "stage0_verify.chk_nonempty_file",
        lambda path, name: type("CheckObj", (), {"name": name, "ok": True, "detail": str(path)})(),
    )

    assert run_all(data.parent, strict=True, check_stage0_flag=False) == 0
    assert run_all(data.parent, strict=True, check_stage0_flag=True) == 1
