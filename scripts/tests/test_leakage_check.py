"""Unit tests for fail-closed leakage audit behaviour."""

from __future__ import annotations

import subprocess
import sys
import os
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
import leakage_check  # noqa: E402
from leakage_check import (  # noqa: E402
    Thresholds,
    audit,
    inspect_direct_exact_reference,
    ligand_max_tanimoto,
    mmseqs_max_seq_id,
    pocket_sucos,
    sealed_discovery_audit,
    validate_sealed_discovery_audit,
    validate_sealed_discovery_audit_sources,
    validate_direct_exact_manifest,
)
from discovery_canonical import current_rdkit_version, discovery_key  # noqa: E402


def _write_direct_manifest(
    tmp_path: Path,
    smiles_rows: list[str],
    *,
    declared_rows: int | None = None,
) -> tuple[Path, Path]:
    direct = tmp_path / "direct_exact_reference.smi"
    identities = [discovery_key(smiles) for smiles in smiles_rows]
    direct.write_text(
        "".join(
            f"{identity.parent_canonical_smiles} {identity.discovery_key_sha256}\n"
            for identity in identities
        )
    )
    aliases = tmp_path / "aliases.parquet"
    aliases.write_bytes(b"sealed aliases")
    canonicalization_audit = tmp_path / "canonicalization_exclusions.jsonl"
    canonicalization_audit.write_text("")
    registry = tmp_path / "source_registry.json"
    registry.write_text("{}\n")
    builder = Path(__file__).resolve().parents[2] / "scripts/build_discovery_alias_map.py"
    outputs = {
        "aliases_parquet_sha256": hashlib.sha256(aliases.read_bytes()).hexdigest(),
        "direct_reference_sha256": hashlib.sha256(direct.read_bytes()).hexdigest(),
    }
    payload = {
        "schema_version": "discovery_alias_map.v1",
        "registry_sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
        "source_manifest_sha256": "0" * 64,
        "source_artifact_sha256": {},
        "outputs": outputs,
        "counts": {
            "alias_rows": 1,
            "direct_reference_rows": declared_rows or len(identities),
        },
        "canonicalization_audit": {
            "path": canonicalization_audit.name,
            "sha256": hashlib.sha256(canonicalization_audit.read_bytes()).hexdigest(),
            "bytes": canonicalization_audit.stat().st_size,
            "rows": 0,
        },
        "policy": {
            "fail_closed": True,
            "max_canonicalization_exclusion_fraction_ppm_per_source": 10_000,
        },
        "canonical_contract": {"pipeline": ["test"]},
        "builder_script_sha256": hashlib.sha256(builder.read_bytes()).hexdigest(),
        "source_registry": registry.name,
        "output_paths": {
            "aliases_parquet": aliases.name,
            "direct_reference": direct.name,
        },
        "output_bytes": {
            "aliases_parquet": aliases.stat().st_size,
            "direct_reference": direct.stat().st_size,
        },
    }
    canonical = {
        key: payload[key]
        for key in (
            "schema_version",
            "registry_sha256",
            "source_manifest_sha256",
            "source_artifact_sha256",
            "outputs",
            "counts",
            "canonicalization_audit",
            "policy",
            "canonical_contract",
            "builder_script_sha256",
        )
    }
    payload["canonical_payload_sha256"] = hashlib.sha256(
        json.dumps(
            canonical,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    payload["binding_sha256"] = leakage_check._payload_binding_sha256(payload)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload) + "\n")
    return direct, manifest


def test_discovery_canonical_key_collapses_salt_and_stereo_variants() -> None:
    base = discovery_key("C[C@H](O)C(=O)[O-].[Na+]")
    variant = discovery_key("CC(O)C(=O)O")

    assert base.parent_canonical_smiles == variant.parent_canonical_smiles
    assert base.discovery_key_sha256 == variant.discovery_key_sha256
    assert len(base.discovery_key_sha256) == 64
    assert base.parent_connectivity_inchikey == base.parent_inchikey.split("-")[0]


def test_direct_exact_manifest_binds_active_reference(tmp_path: Path) -> None:
    direct, manifest = _write_direct_manifest(tmp_path, ["CCO"])

    validate_direct_exact_manifest(direct, manifest)
    changed = discovery_key("CCC")
    direct.write_text(
        f"{changed.parent_canonical_smiles} {changed.discovery_key_sha256}\n"
    )
    with pytest.raises(ValueError, match="reference SHA-256 drift"):
        validate_direct_exact_manifest(direct, manifest)


def test_direct_exact_manifest_rejects_unsorted_rows(tmp_path: Path) -> None:
    direct, manifest = _write_direct_manifest(tmp_path, ["CCO", "CCC"])

    with pytest.raises(ValueError, match="strictly sorted"):
        validate_direct_exact_manifest(direct, manifest)


def test_direct_exact_manifest_rejects_declared_row_count_drift(
    tmp_path: Path,
) -> None:
    direct, manifest = _write_direct_manifest(
        tmp_path,
        ["CCC", "CCO"],
        declared_rows=3,
    )

    with pytest.raises(ValueError, match="row-count drift"):
        validate_direct_exact_manifest(direct, manifest)


def test_direct_exact_reference_rejects_stored_key_mismatch(
    tmp_path: Path,
) -> None:
    direct = tmp_path / "direct_exact_reference.smi"
    direct.write_text(f"CCO {'0' * 64}\n")

    with pytest.raises(ValueError, match="key does not bind its SMILES"):
        inspect_direct_exact_reference(
            direct,
            expected_sha256=hashlib.sha256(direct.read_bytes()).hexdigest(),
            expected_bytes=direct.stat().st_size,
            expected_rows=1,
            stored_keys_out=set(),
        )


def test_leakage_audit_fails_closed_when_axes_missing(tmp_path: Path) -> None:
    rows = pd.DataFrame([{"uniprot": "P12345", "smiles": "CCO"}])
    with pytest.raises(RuntimeError, match="Leakage audit incomplete"):
        audit(
            rows,
            training_seq_db=tmp_path / "missing_seq",
            training_ligands=tmp_path / "missing_ligands.smi",
            training_holo=tmp_path / "missing_holo.tar",
            thresholds=Thresholds(0.30, 0.50, 0.50),
        )


def test_leakage_audit_can_report_incomplete_rows(tmp_path: Path) -> None:
    rows = pd.DataFrame([{"uniprot": "P12345", "smiles": "CCO"}])
    out = audit(
        rows,
        training_seq_db=tmp_path / "missing_seq",
        training_ligands=tmp_path / "missing_ligands.smi",
        training_holo=tmp_path / "missing_holo.tar",
        thresholds=Thresholds(0.30, 0.50, 0.50),
        allow_incomplete=True,
    )
    assert out.loc[0, "audit_status"] == "incomplete"
    assert "sequence" in out.loc[0, "missing_axes"]


def test_leakage_audit_rejects_missing_required_columns(tmp_path: Path) -> None:
    rows = pd.DataFrame([{"uniprot": "P12345"}])
    with pytest.raises(RuntimeError, match="missing required column"):
        audit(
            rows,
            training_seq_db=tmp_path / "seq.fasta",
            training_ligands=tmp_path / "ligands.smi",
            training_holo=tmp_path / "pockets.csv",
            thresholds=Thresholds(0.30, 0.50, 0.50),
        )


def test_leakage_audit_rejects_blank_required_ids(tmp_path: Path) -> None:
    rows = pd.DataFrame([{"uniprot": "P12345", "smiles": "CCO"}, {"uniprot": None, "smiles": "CCN"}])
    with pytest.raises(RuntimeError, match="column 'uniprot' contains blank values"):
        audit(
            rows,
            training_seq_db=tmp_path / "seq.fasta",
            training_ligands=tmp_path / "ligands.smi",
            training_holo=tmp_path / "pockets.csv",
            thresholds=Thresholds(0.30, 0.50, 0.50),
            allow_incomplete=True,
        )


def test_leakage_audit_rejects_duplicate_canonical_eval_pairs(tmp_path: Path) -> None:
    rows = pd.DataFrame([
        {"uniprot": "P12345", "smiles": "C[C@H](O)C(=O)[O-].[Na+]"},
        {"uniprot": "P12345", "smiles": "CC(O)C(=O)O"},
    ])

    with pytest.raises(RuntimeError, match="duplicate canonical evaluation pair"):
        audit(
            rows,
            training_seq_db=tmp_path / "missing_seq",
            training_ligands=tmp_path / "missing_ligands.smi",
            training_holo=tmp_path / "missing_holo.tar",
            thresholds=Thresholds(0.30, 0.50, 0.50),
            allow_incomplete=True,
        )


def test_leakage_audit_rejects_invalid_thresholds(tmp_path: Path) -> None:
    rows = pd.DataFrame([{"uniprot": "P12345", "smiles": "CCO"}])
    with pytest.raises(RuntimeError, match="must be a finite value in \\[0, 1\\]"):
        audit(
            rows,
            training_seq_db=tmp_path / "seq.fasta",
            training_ligands=tmp_path / "ligands.smi",
            training_holo=tmp_path / "pockets.csv",
            thresholds=Thresholds(-0.01, 0.50, 0.50),
            allow_incomplete=True,
        )


def test_leakage_audit_treats_blank_sequence_as_incomplete(tmp_path: Path) -> None:
    seq = tmp_path / "training_cutoff_seqs.fasta"
    seq.write_text(">ref\nMABCDE\n")
    lig = tmp_path / "ligands.smi"
    lig.write_text("CCO ref\n")
    pockets = tmp_path / "pockets.tsv"
    pockets.write_text("target_id\tmax_sucos\nP12345\t0.1\n")
    rows = pd.DataFrame([{
        "uniprot": "P12345",
        "smiles": "CCO",
        "sequence": " ",
    }])

    with pytest.raises(RuntimeError, match="sequence"):
        audit(
            rows,
            training_seq_db=seq,
            training_ligands=lig,
            training_holo=pockets,
            thresholds=Thresholds(0.30, 0.50, 0.50),
        )


def test_leakage_cli_removes_stale_output_on_failure(tmp_path: Path) -> None:
    cases = tmp_path / "cases.csv"
    out = tmp_path / "leakage.csv"
    pd.DataFrame([{"uniprot": "P12345"}]).to_csv(cases, index=False)
    out.write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[2] / "eval/leakage_check.py"),
            "--input-csv", str(cases),
            "--training-seq-db", str(tmp_path / "missing_seq"),
            "--training-ligands", str(tmp_path / "missing_ligands.smi"),
            "--training-holo", str(tmp_path / "missing_holo.tar"),
            "--out-csv", str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "missing required column" in res.stderr
    assert not out.exists()


def test_ligand_axis_rejects_invalid_reference_smiles(tmp_path: Path) -> None:
    lig = tmp_path / "ligands.smi"
    lig.write_text("not_a_smiles ref\n")

    with pytest.raises(RuntimeError, match="invalid SMILES"):
        ligand_max_tanimoto("CCO", lig)

    seq = tmp_path / "training_cutoff_seqs.fasta"
    seq.write_text(">ref\nMABCDE\n")
    pockets = tmp_path / "pockets.tsv"
    pockets.write_text("target_id\tmax_sucos\nP12345\t0.1\n")
    rows = pd.DataFrame([{
        "uniprot": "P12345",
        "smiles": "CCO",
        "sequence": "MABCXE",
    }])
    with pytest.raises(RuntimeError, match="invalid SMILES"):
        audit(
            rows,
            training_seq_db=seq,
            training_ligands=lig,
            training_holo=pockets,
            thresholds=Thresholds(0.30, 0.50, 0.50),
        )


def test_ligand_axis_rejects_duplicate_canonical_reference_smiles(tmp_path: Path) -> None:
    lig = tmp_path / "ligands.smi"
    lig.write_text("CCO ref1\nOCC ref2\n")

    with pytest.raises(RuntimeError, match="duplicate canonical SMILES"):
        ligand_max_tanimoto("CCN", lig)


def test_ligand_axis_rejects_invalid_input_smiles_even_when_incomplete_allowed(
    tmp_path: Path,
) -> None:
    lig = tmp_path / "ligands.smi"
    lig.write_text("CCO ref\n")

    with pytest.raises(RuntimeError, match="input contains invalid SMILES"):
        ligand_max_tanimoto("not_a_smiles", lig)

    rows = pd.DataFrame([{
        "uniprot": "P12345",
        "smiles": "not_a_smiles",
    }])
    with pytest.raises(RuntimeError, match="input contains invalid SMILES"):
        audit(
            rows,
            training_seq_db=tmp_path / "missing_seq",
            training_ligands=lig,
            training_holo=tmp_path / "missing_holo.tar",
            thresholds=Thresholds(0.30, 0.50, 0.50),
            allow_incomplete=True,
        )


def test_ligand_axis_incomplete_when_reference_has_no_rows(tmp_path: Path) -> None:
    lig = tmp_path / "ligands.smi"
    lig.write_text("\n")

    assert ligand_max_tanimoto("CCO", lig) is None

    seq = tmp_path / "training_cutoff_seqs.fasta"
    seq.write_text(">ref\nMABCDE\n")
    pockets = tmp_path / "pockets.tsv"
    pockets.write_text("target_id\tmax_sucos\nP12345\t0.1\n")
    rows = pd.DataFrame([{
        "uniprot": "P12345",
        "smiles": "CCO",
        "sequence": "MABCXE",
    }])
    with pytest.raises(RuntimeError, match="ligand"):
        audit(
            rows,
            training_seq_db=seq,
            training_ligands=lig,
            training_holo=pockets,
            thresholds=Thresholds(0.30, 0.50, 0.50),
        )


def test_sequence_identity_from_cutoff_fasta(tmp_path: Path) -> None:
    fasta = tmp_path / "training_cutoff_seqs.fasta"
    fasta.write_text(">ref1\nMABCDE\n>ref2\nYYYYYY\n")
    score = mmseqs_max_seq_id("P12345", fasta, query_sequence="MABCXE")
    assert score is not None
    assert score >= 5 / 6


def test_sequence_identity_flags_domain_fragment_containment_without_mmseqs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(leakage_check.shutil, "which", lambda _: None)
    fasta = tmp_path / "training_cutoff_seqs.fasta"
    fasta.write_text(">ref1\nXXXXMABCDEYYYY\n")
    score = mmseqs_max_seq_id("P12345", fasta, query_sequence="MABCDE")
    assert score == 1.0


def test_sequence_axis_flags_short_query_contained_in_long_reference_without_mmseqs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(leakage_check.shutil, "which", lambda _: None)
    seq = tmp_path / "training_cutoff_seqs.fasta"
    seq.write_text(">ref\nXXXXMABCDEYYYY\n")
    lig = tmp_path / "ligands.smi"
    lig.write_text("c1ccccc1 ref\n")
    pockets = tmp_path / "pockets.tsv"
    pockets.write_text("target_id\tmax_sucos\nP12345\t0.0\n")
    rows = pd.DataFrame([{
        "uniprot": "P12345",
        "smiles": "CCO",
        "sequence": "MABCDE",
    }])

    out = audit(
        rows,
        training_seq_db=seq,
        training_ligands=lig,
        training_holo=pockets,
        thresholds=Thresholds(0.95, 1.0, 1.0),
    )

    assert out.loc[0, "seq_id"] == 1.0
    assert out.loc[0, "audit_status"] == "ok"
    assert out.loc[0, "leak_flag"]


def test_sequence_axis_uses_mmseqs_when_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_mmseqs = fake_bin / "mmseqs"
    fake_mmseqs.write_text(
        "#!/bin/sh\n"
        "test \"$1\" = easy-search || exit 2\n"
        "grep -q '^>P12345$' \"$2\" || exit 3\n"
        "printf 'P12345\\tref1\\t95.0\\n' > \"$4\"\n"
    )
    fake_mmseqs.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.defpath}")
    fasta = tmp_path / "training_cutoff_seqs.fasta"
    fasta.write_text(">ref1\nMABCDE\n")

    score = mmseqs_max_seq_id("P12345", fasta, query_sequence="MABCXE")

    assert score == 0.95


def test_sequence_axis_fails_closed_when_available_mmseqs_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_mmseqs = fake_bin / "mmseqs"
    fake_mmseqs.write_text("#!/bin/sh\nprintf 'forced failure\\n' >&2\nexit 7\n")
    fake_mmseqs.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.defpath}")
    fasta = tmp_path / "training_cutoff_seqs.fasta"
    fasta.write_text(">ref1\nMABCDE\n")

    with pytest.raises(RuntimeError, match="MMseqs sequence leakage search failed"):
        mmseqs_max_seq_id("P12345", fasta, query_sequence="MABCXE")


def test_sequence_reference_rejects_blank_fasta_header(tmp_path: Path) -> None:
    fasta = tmp_path / "training_cutoff_seqs.fasta"
    fasta.write_text(">\nMABCDE\n")

    with pytest.raises(RuntimeError, match="blank header"):
        mmseqs_max_seq_id("P12345", fasta, query_sequence="MABCXE")


def test_sequence_reference_rejects_duplicate_fasta_header(tmp_path: Path) -> None:
    fasta = tmp_path / "training_cutoff_seqs.fasta"
    fasta.write_text(">ref1\nMABCDE\n>ref1\nYYYYYY\n")

    with pytest.raises(RuntimeError, match="duplicate FASTA header"):
        mmseqs_max_seq_id("P12345", fasta, query_sequence="MABCXE")


def test_sequence_reference_rejects_sequence_before_first_header(tmp_path: Path) -> None:
    fasta = tmp_path / "training_cutoff_seqs.fasta"
    fasta.write_text("MABCDE\n>ref1\nYYYYYY\n")

    with pytest.raises(RuntimeError, match="sequence before first header"):
        mmseqs_max_seq_id("P12345", fasta, query_sequence="MABCXE")


def test_sequence_reference_rejects_empty_fasta_record(tmp_path: Path) -> None:
    fasta = tmp_path / "training_cutoff_seqs.fasta"
    fasta.write_text(">ref1\nMABCDE\n>ref2\n")

    with pytest.raises(RuntimeError, match="has no sequence"):
        mmseqs_max_seq_id("P12345", fasta, query_sequence="MABCXE")


def test_pocket_sucos_from_csv_reference(tmp_path: Path) -> None:
    ref = tmp_path / "pocket_sucos.csv"
    ref.write_text("uniprot,pocket_sucos\nP1,0.25\nP2,0.75\nP2,0.50\n")
    assert pocket_sucos("P2", ref) == 0.75


def test_pocket_sucos_rejects_missing_target_column(tmp_path: Path) -> None:
    ref = tmp_path / "pocket_sucos.csv"
    ref.write_text("protein,pocket_sucos\nP1,0.25\n")

    with pytest.raises(RuntimeError, match="missing target id column"):
        pocket_sucos("P1", ref)


def test_pocket_sucos_rejects_missing_score_column(tmp_path: Path) -> None:
    ref = tmp_path / "pocket_sucos.csv"
    ref.write_text("uniprot,score\nP1,0.25\n")

    with pytest.raises(RuntimeError, match="missing score column"):
        pocket_sucos("P1", ref)


def test_pocket_sucos_rejects_header_only_reference(tmp_path: Path) -> None:
    ref = tmp_path / "pocket_sucos.csv"
    ref.write_text("uniprot,pocket_sucos\n")

    with pytest.raises(RuntimeError, match="contains no rows"):
        pocket_sucos("P1", ref)


def test_pocket_sucos_rejects_nonfinite_csv_reference_score(tmp_path: Path) -> None:
    ref = tmp_path / "pocket_sucos.csv"
    ref.write_text("uniprot,pocket_sucos\nP1,inf\n")

    with pytest.raises(RuntimeError, match="must be a finite value in \\[0, 1\\]"):
        pocket_sucos("P1", ref)


def test_pocket_sucos_rejects_out_of_range_csv_reference_score(tmp_path: Path) -> None:
    ref = tmp_path / "pocket_sucos.csv"
    ref.write_text("uniprot,pocket_sucos\nP1,1.1\n")

    with pytest.raises(RuntimeError, match="must be a finite value in \\[0, 1\\]"):
        pocket_sucos("P1", ref)


def test_pocket_sucos_rejects_unmatched_malformed_csv_reference_score(
    tmp_path: Path,
) -> None:
    ref = tmp_path / "pocket_sucos.csv"
    ref.write_text("uniprot,pocket_sucos\nP1,0.25\nP2,inf\n")

    with pytest.raises(RuntimeError, match="must be a finite value in \\[0, 1\\]"):
        pocket_sucos("P1", ref)


def test_pocket_sucos_preserves_zero_json_score(tmp_path: Path) -> None:
    ref = tmp_path / "pocket_sucos.json"
    ref.write_text('{"P1": {"pocket_sucos": 0.0}}\n')

    assert pocket_sucos("P1", ref) == 0.0


def test_pocket_sucos_rejects_duplicate_json_target(tmp_path: Path) -> None:
    ref = tmp_path / "pocket_sucos.json"
    ref.write_text('{"P1": {"pocket_sucos": 0.1}, "P1": {"pocket_sucos": 0.9}}\n')

    with pytest.raises(RuntimeError, match="duplicate target"):
        pocket_sucos("P1", ref)


def test_pocket_sucos_rejects_non_object_json_reference(tmp_path: Path) -> None:
    ref = tmp_path / "pocket_sucos.json"
    ref.write_text('[{"target_id": "P1", "pocket_sucos": 0.1}]\n')

    with pytest.raises(RuntimeError, match="must be an object"):
        pocket_sucos("P1", ref)


def test_pocket_sucos_rejects_empty_json_reference(tmp_path: Path) -> None:
    ref = tmp_path / "pocket_sucos.json"
    ref.write_text("{}\n")

    with pytest.raises(RuntimeError, match="contains no targets"):
        pocket_sucos("P1", ref)


def test_pocket_sucos_rejects_existing_unsupported_suffix(tmp_path: Path) -> None:
    ref = tmp_path / "training_holo_pockets.tar"
    ref.write_bytes(b"not a supported pocket reference")

    with pytest.raises(RuntimeError, match="unsupported suffix"):
        pocket_sucos("P1", ref)


def test_pocket_sucos_rejects_json_target_without_score(tmp_path: Path) -> None:
    ref = tmp_path / "pocket_sucos.json"
    ref.write_text('{"P1": {"score": 0.1}}\n')

    with pytest.raises(RuntimeError, match="missing score"):
        pocket_sucos("P1", ref)


def test_leakage_audit_ok_when_all_axes_have_references(tmp_path: Path) -> None:
    seq = tmp_path / "training_cutoff_seqs.fasta"
    seq.write_text(">ref\nMABCDE\n")
    lig = tmp_path / "ligands.smi"
    lig.write_text("CCO ref\n")
    pockets = tmp_path / "pockets.tsv"
    pockets.write_text("target_id\tmax_sucos\nP12345\t0.1\n")
    rows = pd.DataFrame([{
        "uniprot": "P12345",
        "smiles": "CCO",
        "sequence": "MABCXE",
    }])

    out = audit(
        rows,
        training_seq_db=seq,
        training_ligands=lig,
        training_holo=pockets,
        thresholds=Thresholds(0.30, 0.50, 0.50),
    )
    assert out.loc[0, "audit_status"] == "ok"
    assert out.loc[0, "leak_flag"]
    assert out.loc[0, "discovery_key_sha256"] == discovery_key("CCO").discovery_key_sha256
    assert out.loc[0, "parent_connectivity_inchikey"]


def test_sealed_discovery_audit_records_hashes_and_survivors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = tmp_path / "cases.csv"
    seq = tmp_path / "training_cutoff_seqs.fasta"
    lig = tmp_path / "ligands.smi"
    pockets = tmp_path / "pockets.csv"
    out_csv = tmp_path / "leakage_audit.csv"
    rows = pd.DataFrame([{"uniprot": "P12345", "smiles": "CCO", "sequence": "MABCXE"}])
    rows.to_csv(cases, index=False)
    seq.write_text(">ref\nYYYYY\n")
    lig.write_text("c1ccccc1 ref\n")
    pockets.write_text("target_id,pocket_sucos\nP12345,0.1\n")
    direct, direct_manifest = _write_direct_manifest(tmp_path, ["CCN"])

    thresholds = Thresholds(0.95, 1.0, 1.0)
    out = audit(
        rows,
        training_seq_db=seq,
        training_ligands=lig,
        training_holo=pockets,
        thresholds=thresholds,
    )
    out.to_csv(out_csv, index=False)
    stored_keys: set[str] = set()
    validate_direct_exact_manifest(
        direct,
        direct_manifest,
        stored_keys_out=stored_keys,
    )
    assert stored_keys == {discovery_key("CCN").discovery_key_sha256}
    monkeypatch.setattr(
        leakage_check,
        "_direct_exact_keys",
        lambda _path: pytest.fail(
            "manifest-backed audit must not re-canonicalize the direct reference"
        ),
    )
    payload = sealed_discovery_audit(
        rows,
        out,
        input_csv=cases,
        training_seq_db=seq,
        training_ligands=lig,
        training_holo=pockets,
        direct_exact_reference=direct,
        out_csv=out_csv,
        thresholds=thresholds,
        execution_challenge="a" * 64,
        manifest_validated_direct_keys=stored_keys,
    )

    assert payload["schema_version"] == "skinscout.discovery-leakage-audit.v2"
    assert payload["status"] == "ok"
    assert payload["direct_exact_count"] == 0
    assert payload["neutral_exclusion_count"] == 0
    assert payload["counts"]["survivor_rows"] == 1
    assert payload["inputs"]["input_csv"]["sha256"] == hashlib.sha256(cases.read_bytes()).hexdigest()
    assert payload["inputs"]["leakage_audit_csv"]["sha256"] == hashlib.sha256(out_csv.read_bytes()).hexdigest()
    assert len(payload["binding_sha256"]) == 64
    validate_sealed_discovery_audit(payload)
    validate_sealed_discovery_audit_sources(
        payload,
        expected_sources={
            "input_csv": cases,
            "training_seq_db": seq,
            "training_ligands": lig,
            "training_holo": pockets,
            "direct_exact_reference": direct,
            "leakage_audit_csv": out_csv,
        },
        expected_thresholds=thresholds,
        expected_execution_challenge="a" * 64,
        require_direct_exact_reference=True,
        direct_exact_manifest=direct_manifest,
    )


def test_sealed_discovery_audit_fails_when_direct_exact_reference_matches(
    tmp_path: Path,
) -> None:
    cases = tmp_path / "cases.csv"
    seq = tmp_path / "training_cutoff_seqs.fasta"
    lig = tmp_path / "ligands.smi"
    pockets = tmp_path / "pockets.csv"
    direct = tmp_path / "direct.smi"
    out_csv = tmp_path / "leakage_audit.csv"
    rows = pd.DataFrame([{"uniprot": "P12345", "smiles": "CCO", "sequence": "MABCXE"}])
    rows.to_csv(cases, index=False)
    seq.write_text(">ref\nYYYYY\n")
    lig.write_text("c1ccccc1 ref\n")
    pockets.write_text("target_id,pocket_sucos\nP12345,0.1\n")
    direct.write_text("OCC direct\n")

    thresholds = Thresholds(0.95, 1.0, 1.0)
    out = audit(
        rows,
        training_seq_db=seq,
        training_ligands=lig,
        training_holo=pockets,
        thresholds=thresholds,
    )
    out.to_csv(out_csv, index=False)
    payload = sealed_discovery_audit(
        rows,
        out,
        input_csv=cases,
        training_seq_db=seq,
        training_ligands=lig,
        training_holo=pockets,
        direct_exact_reference=direct,
        out_csv=out_csv,
        thresholds=thresholds,
        execution_challenge="b" * 64,
    )

    assert payload["status"] == "failed"
    assert payload["direct_exact_count"] == 1
    assert payload["neutral_exclusion_count"] == 1
    assert payload["excluded_rows"][0]["reasons"] == ["direct_exact_match"]


def test_sealed_discovery_audit_allows_axis_exclusion_when_survivor_remains(
    tmp_path: Path,
) -> None:
    cases = tmp_path / "cases.csv"
    seq = tmp_path / "training_cutoff_seqs.fasta"
    lig = tmp_path / "ligands.smi"
    pockets = tmp_path / "pockets.csv"
    direct = tmp_path / "direct.smi"
    out_csv = tmp_path / "leakage_audit.csv"
    rows = pd.DataFrame(
        [
            {"uniprot": "P1", "smiles": "CCO"},
            {"uniprot": "P2", "smiles": "CCC"},
        ]
    )
    rows.to_csv(cases, index=False)
    seq.write_text(">ref\nYYYY\n")
    lig.write_text("c1ccccc1 ref\n")
    pockets.write_text("target_id,pocket_sucos\nP1,0.1\nP2,0.1\n")
    direct.write_text("CCN direct\n")
    output_rows = []
    for uniprot, smiles, seq_id in (("P1", "CCO", 0.9), ("P2", "CCC", 0.1)):
        key = discovery_key(smiles)
        output_rows.append(
            {
                "uniprot": uniprot,
                "smiles": smiles,
                "parent_canonical_smiles": key.parent_canonical_smiles,
                "discovery_key_sha256": key.discovery_key_sha256,
                "parent_inchikey": key.parent_inchikey,
                "parent_connectivity_inchikey": key.parent_connectivity_inchikey,
                "seq_id": seq_id,
                "ligand_tanimoto": 0.1,
                "pocket_sucos": 0.1,
                "leak_flag": seq_id >= 0.5,
                "audit_status": "ok",
                "missing_axes": "",
            }
        )
    out = pd.DataFrame(output_rows)
    out.to_csv(out_csv, index=False)
    thresholds = Thresholds(0.5, 0.5, 0.5)

    payload = sealed_discovery_audit(
        rows,
        out,
        input_csv=cases,
        training_seq_db=seq,
        training_ligands=lig,
        training_holo=pockets,
        direct_exact_reference=direct,
        out_csv=out_csv,
        thresholds=thresholds,
        execution_challenge="1" * 64,
    )

    assert payload["status"] == "ok"
    assert payload["counts"]["survivor_rows"] == 1
    assert payload["excluded_rows"][0]["reasons"] == ["leakage_axis_flag"]
    validate_sealed_discovery_audit_sources(
        payload,
        expected_sources={
            "input_csv": cases,
            "training_seq_db": seq,
            "training_ligands": lig,
            "training_holo": pockets,
            "direct_exact_reference": direct,
            "leakage_audit_csv": out_csv,
        },
        expected_thresholds=thresholds,
        expected_execution_challenge="1" * 64,
        require_direct_exact_reference=True,
    )


def test_direct_exact_reference_missing_or_empty_fails_when_supplied(
    tmp_path: Path,
) -> None:
    rows = pd.DataFrame([{"uniprot": "P12345", "smiles": "CCO", "sequence": "MABCXE"}])
    out = pd.DataFrame([{
        "uniprot": "P12345",
        "smiles": "CCO",
        "parent_canonical_smiles": discovery_key("CCO").parent_canonical_smiles,
        "discovery_key_sha256": discovery_key("CCO").discovery_key_sha256,
        "parent_inchikey": discovery_key("CCO").parent_inchikey,
        "parent_connectivity_inchikey": discovery_key("CCO").parent_connectivity_inchikey,
        "leak_flag": False,
        "audit_status": "ok",
    }])
    for direct in (tmp_path / "missing.smi", tmp_path / "empty.smi"):
        if direct.name == "empty.smi":
            direct.write_text("")
        with pytest.raises(RuntimeError, match="Direct exact reference"):
            sealed_discovery_audit(
                rows,
                out,
                input_csv=tmp_path / "cases.csv",
                training_seq_db=tmp_path / "seq.fasta",
                training_ligands=tmp_path / "ligands.smi",
                training_holo=tmp_path / "pockets.csv",
                direct_exact_reference=direct,
                out_csv=tmp_path / "leakage_audit.csv",
                thresholds=Thresholds(0.95, 1.0, 1.0),
                execution_challenge="c" * 64,
            )


def test_sealed_discovery_audit_validator_rejects_minimal_handwritten_payload() -> None:
    with pytest.raises(ValueError, match="RDKit"):
        validate_sealed_discovery_audit({
            "schema_version": "skinscout.discovery-leakage-audit.v2",
            "status": "ok",
            "direct_exact_count": 0,
            "counts": {"survivor_rows": 1},
            "survivors": [{"discovery_key_sha256": "a" * 64}],
        })


def test_sealed_discovery_audit_validator_rejects_cross_runtime_rdkit_drift(
    tmp_path: Path,
) -> None:
    cases = tmp_path / "cases.csv"
    seq = tmp_path / "training_cutoff_seqs.fasta"
    lig = tmp_path / "ligands.smi"
    pockets = tmp_path / "pockets.csv"
    direct = tmp_path / "direct.smi"
    out_csv = tmp_path / "leakage_audit.csv"
    rows = pd.DataFrame([{"uniprot": "P12345", "smiles": "CCO", "sequence": "MABCXE"}])
    rows.to_csv(cases, index=False)
    seq.write_text(">ref\nYYYYY\n")
    lig.write_text("c1ccccc1 ref\n")
    pockets.write_text("target_id,pocket_sucos\nP12345,0.1\n")
    direct.write_text("CCN direct\n")
    thresholds = Thresholds(0.95, 1.0, 1.0)
    out = audit(
        rows,
        training_seq_db=seq,
        training_ligands=lig,
        training_holo=pockets,
        thresholds=thresholds,
    )
    out.to_csv(out_csv, index=False)
    payload = sealed_discovery_audit(
        rows,
        out,
        input_csv=cases,
        training_seq_db=seq,
        training_ligands=lig,
        training_holo=pockets,
        direct_exact_reference=direct,
        out_csv=out_csv,
        thresholds=thresholds,
        execution_challenge="d" * 64,
    )
    payload["rdkit_version"] = "0.0.0"

    assert current_rdkit_version() != "0.0.0"
    with pytest.raises(ValueError, match="active RDKit runtime"):
        validate_sealed_discovery_audit(payload)


def test_sealed_discovery_audit_rejects_changed_active_source(tmp_path: Path) -> None:
    cases = tmp_path / "cases.csv"
    seq = tmp_path / "training_cutoff_seqs.fasta"
    lig = tmp_path / "ligands.smi"
    pockets = tmp_path / "pockets.csv"
    direct = tmp_path / "direct.smi"
    out_csv = tmp_path / "leakage_audit.csv"
    rows = pd.DataFrame([{"uniprot": "P12345", "smiles": "CCO", "sequence": "MABCXE"}])
    rows.to_csv(cases, index=False)
    seq.write_text(">ref\nYYYYY\n")
    lig.write_text("c1ccccc1 ref\n")
    pockets.write_text("target_id,pocket_sucos\nP12345,0.1\n")
    direct.write_text("CCN direct\n")
    thresholds = Thresholds(0.95, 1.0, 1.0)
    out = audit(rows, seq, lig, pockets, thresholds)
    out.to_csv(out_csv, index=False)
    payload = sealed_discovery_audit(
        rows,
        out,
        input_csv=cases,
        training_seq_db=seq,
        training_ligands=lig,
        training_holo=pockets,
        direct_exact_reference=direct,
        out_csv=out_csv,
        thresholds=thresholds,
        execution_challenge="e" * 64,
    )

    direct.write_text("CCC changed\n")

    with pytest.raises(ValueError, match="active source record mismatch"):
        validate_sealed_discovery_audit_sources(
            payload,
            require_direct_exact_reference=True,
        )


def test_sealed_discovery_audit_rejects_binding_tamper(tmp_path: Path) -> None:
    cases = tmp_path / "cases.csv"
    seq = tmp_path / "training_cutoff_seqs.fasta"
    lig = tmp_path / "ligands.smi"
    pockets = tmp_path / "pockets.csv"
    direct = tmp_path / "direct.smi"
    out_csv = tmp_path / "leakage_audit.csv"
    rows = pd.DataFrame([{"uniprot": "P12345", "smiles": "CCO", "sequence": "MABCXE"}])
    rows.to_csv(cases, index=False)
    seq.write_text(">ref\nYYYYY\n")
    lig.write_text("c1ccccc1 ref\n")
    pockets.write_text("target_id,pocket_sucos\nP12345,0.1\n")
    direct.write_text("CCN direct\n")
    thresholds = Thresholds(0.95, 1.0, 1.0)
    out = audit(rows, seq, lig, pockets, thresholds)
    out.to_csv(out_csv, index=False)
    payload = sealed_discovery_audit(
        rows,
        out,
        input_csv=cases,
        training_seq_db=seq,
        training_ligands=lig,
        training_holo=pockets,
        direct_exact_reference=direct,
        out_csv=out_csv,
        thresholds=thresholds,
        execution_challenge="f" * 64,
    )
    payload["counts"]["survivor_rows"] = 0

    with pytest.raises(ValueError, match="binding_sha256"):
        validate_sealed_discovery_audit(payload)


def test_sealed_discovery_audit_sources_reject_forged_score_flag_mismatch(
    tmp_path: Path,
) -> None:
    cases = tmp_path / "cases.csv"
    seq = tmp_path / "training_cutoff_seqs.fasta"
    lig = tmp_path / "ligands.smi"
    pockets = tmp_path / "pockets.csv"
    direct = tmp_path / "direct.smi"
    out_csv = tmp_path / "leakage_audit.csv"
    rows = pd.DataFrame([{"uniprot": "P12345", "smiles": "CCO", "sequence": "MABCXE"}])
    rows.to_csv(cases, index=False)
    seq.write_text(">ref\nYYYYY\n")
    lig.write_text("c1ccccc1 ref\n")
    pockets.write_text("target_id,pocket_sucos\nP12345,0.1\n")
    direct.write_text("CCN direct\n")
    thresholds = Thresholds(0.95, 1.0, 1.0)
    out = audit(rows, seq, lig, pockets, thresholds)
    out.to_csv(out_csv, index=False)
    payload = sealed_discovery_audit(
        rows,
        out,
        input_csv=cases,
        training_seq_db=seq,
        training_ligands=lig,
        training_holo=pockets,
        direct_exact_reference=direct,
        out_csv=out_csv,
        thresholds=thresholds,
        execution_challenge="9" * 64,
    )

    forged = out.copy()
    forged.loc[0, "seq_id"] = 1.0
    forged.loc[0, "leak_flag"] = False
    forged.to_csv(out_csv, index=False)
    payload["inputs"]["leakage_audit_csv"] = leakage_check._source_record(out_csv)
    payload["binding_sha256"] = leakage_check._payload_binding_sha256(payload)

    with pytest.raises(ValueError, match="leak_flag mismatch"):
        validate_sealed_discovery_audit_sources(
            payload,
            expected_sources={
                "input_csv": cases,
                "training_seq_db": seq,
                "training_ligands": lig,
                "training_holo": pockets,
                "direct_exact_reference": direct,
                "leakage_audit_csv": out_csv,
            },
            expected_thresholds=thresholds,
            expected_execution_challenge="9" * 64,
            require_direct_exact_reference=True,
        )


def test_cli_requires_direct_exact_reference_for_sealed_audit(tmp_path: Path) -> None:
    cases = tmp_path / "cases.csv"
    seq = tmp_path / "training_cutoff_seqs.fasta"
    lig = tmp_path / "ligands.smi"
    pockets = tmp_path / "pockets.csv"
    out_csv = tmp_path / "leakage_audit.csv"
    out_json = tmp_path / "discovery_leakage_audit.json"
    pd.DataFrame([{
        "uniprot": "P12345",
        "smiles": "CCO",
        "sequence": "MABCXE",
    }]).to_csv(cases, index=False)
    seq.write_text(">ref\nYYYYY\n")
    lig.write_text("c1ccccc1 ref\n")
    pockets.write_text("target_id,pocket_sucos\nP12345,0.1\n")

    res = subprocess.run(
        [
            sys.executable,
            "eval/leakage_check.py",
            "--input-csv", str(cases),
            "--training-seq-db", str(seq),
            "--training-ligands", str(lig),
            "--training-holo", str(pockets),
            "--seq-id-threshold", "0.95",
            "--ligand-tanimoto-threshold", "1.0",
            "--pocket-sucos-threshold", "1.0",
            "--out-csv", str(out_csv),
            "--out-discovery-audit-json", str(out_json),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "--direct-exact-reference is required" in res.stderr
    assert not out_json.exists()


def test_cli_persists_failed_direct_exact_audit(tmp_path: Path) -> None:
    cases = tmp_path / "cases.csv"
    seq = tmp_path / "training_cutoff_seqs.fasta"
    lig = tmp_path / "ligands.smi"
    pockets = tmp_path / "pockets.csv"
    direct = tmp_path / "direct.smi"
    out_csv = tmp_path / "leakage_audit.csv"
    out_json = tmp_path / "discovery_leakage_audit.json"
    pd.DataFrame([{
        "uniprot": "P12345",
        "smiles": "CCO",
        "sequence": "MABCXE",
    }]).to_csv(cases, index=False)
    seq.write_text(">ref\nYYYYY\n")
    lig.write_text("c1ccccc1 ref\n")
    pockets.write_text("target_id,pocket_sucos\nP12345,0.1\n")
    direct.write_text("OCC exact\n")

    res = subprocess.run(
        [
            sys.executable,
            "eval/leakage_check.py",
            "--input-csv", str(cases),
            "--training-seq-db", str(seq),
            "--training-ligands", str(lig),
            "--training-holo", str(pockets),
            "--direct-exact-reference", str(direct),
            "--execution-challenge", "1" * 64,
            "--seq-id-threshold", "0.95",
            "--ligand-tanimoto-threshold", "1.0",
            "--pocket-sucos-threshold", "1.0",
            "--out-csv", str(out_csv),
            "--out-discovery-audit-json", str(out_json),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "direct_exact_count=1" in res.stderr
    payload = json.loads(out_json.read_text())
    assert payload["status"] == "failed"
    assert payload["direct_exact_count"] == 1
    assert payload["inputs"]["leakage_audit_csv"]["sha256"] == hashlib.sha256(
        out_csv.read_bytes()
    ).hexdigest()
