"""Regression tests for Stage 3 Daina-Zoete fail-closed reference data gates."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import json
import numpy as np
import pandas as pd
import pytest
from rdkit import Chem, DataStructs
from rdkit.Chem import inchi, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from stage3_daina_zoete import fp_from_uint64_row  # noqa: E402


def write_input_sdf(path: Path) -> None:
    mol = Chem.MolFromSmiles("CCO")
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()


def run_daina(
    tmp_path: Path,
    chembl_fp: Path,
    *extra_args: str,
) -> subprocess.CompletedProcess[str]:
    input_sdf = tmp_path / "ligand.sdf"
    write_input_sdf(input_sdf)
    (tmp_path / "daina.tsv").write_text("stale\n")
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_daina_zoete.py"),
            "--ligand-sdf",
            str(input_sdf),
            "--chembl-fp",
            str(chembl_fp),
            "--out-scores",
            str(tmp_path / "daina.tsv"),
            *extra_args,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def fingerprint_words(smiles: str) -> list[int]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprint = generator.GetFingerprint(Chem.MolFromSmiles(smiles))
    bits = np.fromiter((int(bit) for bit in fingerprint.ToBitString()), dtype=np.uint8)
    return np.packbits(bits).view(np.uint64).tolist()


def standard_inchikey(smiles: str) -> str:
    parent = rdMolStandardize.FragmentParent(Chem.MolFromSmiles(smiles))
    return inchi.MolToInchiKey(parent)


@pytest.mark.parametrize("smiles", ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"])
def test_uint64_fingerprint_round_trip_is_bit_exact(smiles: str) -> None:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    expected = generator.GetFingerprint(Chem.MolFromSmiles(smiles))

    restored = fp_from_uint64_row(fingerprint_words(smiles))

    assert restored.ToBitString() == expected.ToBitString()
    assert DataStructs.TanimotoSimilarity(restored, expected) == 1.0


def test_missing_fingerprint_reference_fails_without_empty_output(tmp_path: Path) -> None:
    res = run_daina(tmp_path, tmp_path / "missing.parquet")

    assert res.returncode != 0
    assert "ChEMBL fingerprint parquet is required" in res.stderr
    assert not (tmp_path / "daina.tsv").exists()


def test_empty_fingerprint_reference_fails_without_empty_output(tmp_path: Path) -> None:
    chembl_fp = tmp_path / "fp_morgan2_2048.parquet"
    pd.DataFrame(columns=["molecule_chembl_id", "bitvec"]).to_parquet(chembl_fp)
    pd.DataFrame(
        [{"molecule_chembl_id": "CHEMBL1", "uniprot": "P12345"}]
    ).to_parquet(tmp_path / "human_activities.parquet")

    res = run_daina(tmp_path, chembl_fp)

    assert res.returncode != 0
    assert "contains no ligands" in res.stderr
    assert not (tmp_path / "daina.tsv").exists()


def test_missing_fingerprint_columns_fail_without_empty_output(tmp_path: Path) -> None:
    chembl_fp = tmp_path / "fp_morgan2_2048.parquet"
    pd.DataFrame([{"molecule_chembl_id": "CHEMBL1"}]).to_parquet(chembl_fp)
    pd.DataFrame(
        [{"molecule_chembl_id": "CHEMBL1", "uniprot": "P12345"}]
    ).to_parquet(tmp_path / "human_activities.parquet")

    res = run_daina(tmp_path, chembl_fp)

    assert res.returncode != 0
    assert "missing required columns" in res.stderr
    assert not (tmp_path / "daina.tsv").exists()


def test_malformed_fingerprint_bitvec_fails_without_empty_output(tmp_path: Path) -> None:
    chembl_fp = tmp_path / "fp_morgan2_2048.parquet"
    pd.DataFrame(
        [{"molecule_chembl_id": "CHEMBL1", "bitvec": [0] * 31}]
    ).to_parquet(chembl_fp)
    pd.DataFrame(
        [{"molecule_chembl_id": "CHEMBL1", "uniprot": "P12345"}]
    ).to_parquet(tmp_path / "human_activities.parquet")

    res = run_daina(tmp_path, chembl_fp)

    assert res.returncode != 0
    assert "invalid 2048-bit bitvec" in res.stderr
    assert not (tmp_path / "daina.tsv").exists()


def test_blank_fingerprint_molecule_id_fails_without_empty_output(tmp_path: Path) -> None:
    chembl_fp = tmp_path / "fp_morgan2_2048.parquet"
    pd.DataFrame(
        [{"molecule_chembl_id": " ", "bitvec": [0] * 32}]
    ).to_parquet(chembl_fp)
    pd.DataFrame(
        [{"molecule_chembl_id": "CHEMBL1", "uniprot": "P12345"}]
    ).to_parquet(tmp_path / "human_activities.parquet")

    res = run_daina(tmp_path, chembl_fp)

    assert res.returncode != 0
    assert (
        "ChEMBL fingerprint parquet column 'molecule_chembl_id' contains blank values"
        in res.stderr
    )
    assert not (tmp_path / "daina.tsv").exists()


def test_duplicate_fingerprint_molecule_id_fails_without_order_dependent_scores(
    tmp_path: Path,
) -> None:
    chembl_fp = tmp_path / "fp_morgan2_2048.parquet"
    pd.DataFrame([
        {"molecule_chembl_id": "CHEMBL1", "bitvec": [0] * 32},
        {"molecule_chembl_id": "CHEMBL1", "bitvec": [1] + [0] * 31},
    ]).to_parquet(chembl_fp)
    pd.DataFrame(
        [{"molecule_chembl_id": "CHEMBL1", "uniprot": "P12345"}]
    ).to_parquet(tmp_path / "human_activities.parquet")

    res = run_daina(tmp_path, chembl_fp)

    assert res.returncode != 0
    assert "duplicate molecule_chembl_id values" in res.stderr
    assert "CHEMBL1" in res.stderr
    assert not (tmp_path / "daina.tsv").exists()


def test_decimal_string_uint64_fingerprint_words_are_accepted(
    tmp_path: Path,
) -> None:
    chembl_fp = tmp_path / "fp_morgan2_2048.parquet"
    pd.DataFrame(
        [
            {
                "molecule_chembl_id": "CHEMBL1",
                "bitvec": [str(2**63)] + ["0"] * 31,
            }
        ]
    ).to_parquet(chembl_fp)
    pd.DataFrame(
        [{"molecule_chembl_id": "CHEMBL1", "uniprot": "P12345"}]
    ).to_parquet(tmp_path / "human_activities.parquet")

    res = run_daina(tmp_path, chembl_fp)

    assert res.returncode == 0, res.stderr
    assert "P12345" in (tmp_path / "daina.tsv").read_text()


def test_missing_activity_columns_fail_without_empty_output(tmp_path: Path) -> None:
    chembl_fp = tmp_path / "fp_morgan2_2048.parquet"
    pd.DataFrame(
        [{"molecule_chembl_id": "CHEMBL1", "bitvec": [0] * 32}]
    ).to_parquet(chembl_fp)
    pd.DataFrame([{"molecule_chembl_id": "CHEMBL1"}]).to_parquet(
        tmp_path / "human_activities.parquet"
    )

    res = run_daina(tmp_path, chembl_fp)

    assert res.returncode != 0
    assert "ChEMBL human activities parquet missing required columns" in res.stderr
    assert not (tmp_path / "daina.tsv").exists()


@pytest.mark.parametrize(
    ("column", "message"),
    [
        (
            "molecule_chembl_id",
            "ChEMBL human activities column 'molecule_chembl_id' contains blank values",
        ),
        ("uniprot", "ChEMBL human activities column 'uniprot' contains blank values"),
    ],
)
def test_blank_activity_fields_fail_without_empty_output(
    tmp_path: Path,
    column: str,
    message: str,
) -> None:
    chembl_fp = tmp_path / "fp_morgan2_2048.parquet"
    pd.DataFrame(
        [{"molecule_chembl_id": "CHEMBL1", "bitvec": [0] * 32}]
    ).to_parquet(chembl_fp)
    row = {"molecule_chembl_id": "CHEMBL1", "uniprot": "P12345"}
    row[column] = " "
    pd.DataFrame([row]).to_parquet(tmp_path / "human_activities.parquet")

    res = run_daina(tmp_path, chembl_fp)

    assert res.returncode != 0
    assert message in res.stderr
    assert not (tmp_path / "daina.tsv").exists()


def test_empty_activity_edges_fail_without_empty_output(tmp_path: Path) -> None:
    chembl_fp = tmp_path / "fp_morgan2_2048.parquet"
    pd.DataFrame(
        [{"molecule_chembl_id": "CHEMBL1", "bitvec": [0] * 32}]
    ).to_parquet(chembl_fp)
    pd.DataFrame(columns=["molecule_chembl_id", "uniprot"]).to_parquet(
        tmp_path / "human_activities.parquet"
    )

    res = run_daina(tmp_path, chembl_fp)

    assert res.returncode != 0
    assert "contain no molecule-target edges" in res.stderr
    assert not (tmp_path / "daina.tsv").exists()


def test_duplicate_activity_pair_is_deduplicated_before_scoring(
    tmp_path: Path,
) -> None:
    chembl_fp = tmp_path / "fp_morgan2_2048.parquet"
    pd.DataFrame(
        [{"molecule_chembl_id": "CHEMBL1", "bitvec": [0] * 32}]
    ).to_parquet(chembl_fp)
    pd.DataFrame([
        {"molecule_chembl_id": "CHEMBL1", "uniprot": "P12345"},
        {"molecule_chembl_id": "CHEMBL1", "uniprot": "P12345"},
    ]).to_parquet(tmp_path / "human_activities.parquet")

    res = run_daina(tmp_path, chembl_fp)

    assert res.returncode == 0, res.stderr
    assert "Dropping 1 duplicate ChEMBL molecule-target activity edge" in res.stderr
    scores = pd.read_csv(tmp_path / "daina.tsv", sep="\t")
    assert scores["target_id"].tolist() == ["P12345"]


def test_disjoint_fingerprint_activity_ids_fail_without_zero_scores(tmp_path: Path) -> None:
    chembl_fp = tmp_path / "fp_morgan2_2048.parquet"
    pd.DataFrame(
        [{"molecule_chembl_id": "CHEMBL_WITH_FP", "bitvec": [0] * 32}]
    ).to_parquet(chembl_fp)
    pd.DataFrame(
        [{"molecule_chembl_id": "CHEMBL_ACTIVITY_ONLY", "uniprot": "P12345"}]
    ).to_parquet(tmp_path / "human_activities.parquet")

    res = run_daina(tmp_path, chembl_fp)

    assert res.returncode != 0
    assert "no overlapping molecule IDs" in res.stderr
    assert not (tmp_path / "daina.tsv").exists()


def test_partial_missing_fingerprint_activity_ids_fail_without_partial_scores(
    tmp_path: Path,
) -> None:
    chembl_fp = tmp_path / "fp_morgan2_2048.parquet"
    pd.DataFrame(
        [{"molecule_chembl_id": "CHEMBL_WITH_FP", "bitvec": [0] * 32}]
    ).to_parquet(chembl_fp)
    pd.DataFrame([
        {"molecule_chembl_id": "CHEMBL_WITH_FP", "uniprot": "P12345"},
        {"molecule_chembl_id": "CHEMBL_ACTIVITY_ONLY", "uniprot": "P67890"},
    ]).to_parquet(tmp_path / "human_activities.parquet")

    res = run_daina(tmp_path, chembl_fp)

    assert res.returncode != 0
    assert "activity edges without fingerprints" in res.stderr
    assert "CHEMBL_ACTIVITY_ONLY" in res.stderr
    assert not (tmp_path / "daina.tsv").exists()


def test_leave_query_out_excludes_exact_reference_and_writes_audit_metadata(
    tmp_path: Path,
) -> None:
    chembl_fp = tmp_path / "fp_morgan2_2048.parquet"
    pd.DataFrame([
        {"molecule_chembl_id": "CHEMBL_EXACT", "bitvec": fingerprint_words("CCO")},
        {"molecule_chembl_id": "CHEMBL_OTHER", "bitvec": fingerprint_words("c1ccccc1")},
    ]).to_parquet(chembl_fp)
    pd.DataFrame([
        {
            "molecule_chembl_id": "CHEMBL_EXACT",
            "uniprot": "P_EXACT",
            "standard_inchi_key": standard_inchikey("CCO"),
        },
        {
            "molecule_chembl_id": "CHEMBL_OTHER",
            "uniprot": "P_OTHER",
            "standard_inchi_key": standard_inchikey("c1ccccc1"),
        },
    ]).to_parquet(tmp_path / "human_activities.parquet")
    metadata = tmp_path / "daina.metadata.json"

    res = run_daina(
        tmp_path,
        chembl_fp,
        "--evidence-mode",
        "leave-query-out",
        "--evidence-snapshot-id",
        "chembl-test-sha256",
        "--exclude-reference-similarity",
        "0.99",
        "--out-metadata-json",
        str(metadata),
    )

    assert res.returncode == 0, res.stderr
    scores = pd.read_csv(tmp_path / "daina.tsv", sep="\t")
    assert scores["target_id"].tolist() == ["P_OTHER"]
    payload = json.loads(metadata.read_text())
    assert payload["evidence_mode"] == "leave-query-out"
    assert payload["score_is_calibrated_probability"] is False
    assert payload["counts"]["excluded_reference_molecules_by_similarity"] == 1


def test_temporal_mode_uses_only_dated_pre_cutoff_evidence(tmp_path: Path) -> None:
    chembl_fp = tmp_path / "fp_morgan2_2048.parquet"
    pd.DataFrame([
        {"molecule_chembl_id": "CHEMBL_OLD", "bitvec": fingerprint_words("c1ccccc1")},
        {"molecule_chembl_id": "CHEMBL_FUTURE", "bitvec": fingerprint_words("CCN")},
        {"molecule_chembl_id": "CHEMBL_UNDATED", "bitvec": fingerprint_words("CCC")},
    ]).to_parquet(chembl_fp)
    pd.DataFrame([
        {
            "molecule_chembl_id": "CHEMBL_OLD",
            "uniprot": "P_OLD",
            "evidence_date": "2022-01-02",
            "standard_inchi_key": standard_inchikey("c1ccccc1"),
        },
        {
            "molecule_chembl_id": "CHEMBL_FUTURE",
            "uniprot": "P_FUTURE",
            "evidence_date": "2024-01-02",
            "standard_inchi_key": standard_inchikey("CCN"),
        },
        {
            "molecule_chembl_id": "CHEMBL_UNDATED",
            "uniprot": "P_UNDATED",
            "evidence_date": None,
            "standard_inchi_key": standard_inchikey("CCC"),
        },
    ]).to_parquet(tmp_path / "human_activities.parquet")
    metadata = tmp_path / "daina.metadata.json"

    res = run_daina(
        tmp_path,
        chembl_fp,
        "--evidence-mode",
        "temporal",
        "--evidence-snapshot-id",
        "chembl-pre-2023",
        "--cutoff-date",
        "2023-10-01",
        "--out-metadata-json",
        str(metadata),
    )

    assert res.returncode == 0, res.stderr
    scores = pd.read_csv(tmp_path / "daina.tsv", sep="\t")
    assert scores["target_id"].tolist() == ["P_OLD"]
    payload = json.loads(metadata.read_text())
    assert payload["cutoff_date"] == "2023-10-01"
    assert payload["temporal_filter"]["undated_excluded"] == 1


@pytest.mark.parametrize(
    "missing_arg",
    ["snapshot", "metadata"],
)
def test_leakage_controlled_mode_requires_auditable_run_metadata(
    tmp_path: Path,
    missing_arg: str,
) -> None:
    chembl_fp = tmp_path / "fp_morgan2_2048.parquet"
    pd.DataFrame([
        {"molecule_chembl_id": "CHEMBL1", "bitvec": fingerprint_words("c1ccccc1")},
    ]).to_parquet(chembl_fp)
    pd.DataFrame([
        {
            "molecule_chembl_id": "CHEMBL1",
            "uniprot": "P12345",
            "standard_inchi_key": standard_inchikey("c1ccccc1"),
        },
    ]).to_parquet(tmp_path / "human_activities.parquet")
    extra = ["--evidence-mode", "leave-query-out"]
    if missing_arg != "snapshot":
        extra += ["--evidence-snapshot-id", "snapshot"]
    if missing_arg != "metadata":
        extra += ["--out-metadata-json", str(tmp_path / "daina.metadata.json")]

    res = run_daina(tmp_path, chembl_fp, *extra)

    assert res.returncode != 0
    assert "requires" in res.stderr
    assert not (tmp_path / "daina.tsv").exists()


def test_leave_query_out_excludes_same_connectivity_across_protonation_states(
    tmp_path: Path,
) -> None:
    input_sdf = tmp_path / "ligand.sdf"
    writer = Chem.SDWriter(str(input_sdf))
    writer.write(Chem.MolFromSmiles("CC(=O)O"))
    writer.close()
    chembl_fp = tmp_path / "fp_morgan2_2048.parquet"
    pd.DataFrame([
        {
            "molecule_chembl_id": "CHEMBL_ACETATE",
            "bitvec": fingerprint_words("CC(=O)[O-]"),
        },
        {
            "molecule_chembl_id": "CHEMBL_OTHER",
            "bitvec": fingerprint_words("c1ccccc1"),
        },
    ]).to_parquet(chembl_fp)
    pd.DataFrame([
        {
            "molecule_chembl_id": "CHEMBL_ACETATE",
            "uniprot": "P_SAME",
            "standard_inchi_key": standard_inchikey("CC(=O)[O-]"),
        },
        {
            "molecule_chembl_id": "CHEMBL_OTHER",
            "uniprot": "P_OTHER",
            "standard_inchi_key": standard_inchikey("c1ccccc1"),
        },
    ]).to_parquet(tmp_path / "human_activities.parquet")
    metadata = tmp_path / "daina.metadata.json"

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_daina_zoete.py"),
            "--ligand-sdf",
            str(input_sdf),
            "--chembl-fp",
            str(chembl_fp),
            "--out-scores",
            str(tmp_path / "daina.tsv"),
            "--out-metadata-json",
            str(metadata),
            "--evidence-mode",
            "leave-query-out",
            "--evidence-snapshot-id",
            "fixture",
            "--exclude-reference-similarity",
            "1.0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode == 0, res.stderr
    scores = pd.read_csv(tmp_path / "daina.tsv", sep="\t")
    assert scores["target_id"].tolist() == ["P_OTHER"]
    payload = json.loads(metadata.read_text())
    assert payload["counts"]["excluded_reference_molecules_by_identity"] == 1
    assert payload["counts"]["excluded_reference_molecules_by_similarity"] == 0
    assert payload["counts"]["excluded_reference_molecules_total"] == 1


def test_high_confidence_quality_policy_filters_weak_rows_and_scores_hybrid(
    tmp_path: Path,
) -> None:
    chembl_fp = tmp_path / "fp_morgan2_2048.parquet"
    pd.DataFrame([
        {"molecule_chembl_id": "CHEMBL_GOOD", "bitvec": fingerprint_words("CCN")},
        {"molecule_chembl_id": "CHEMBL_WEAK", "bitvec": fingerprint_words("c1ccccc1")},
    ]).to_parquet(chembl_fp)
    common = {
        "standard_relation": "=",
        "potential_duplicate": False,
        "data_validity_comment": None,
    }
    pd.DataFrame([
        {
            **common,
            "molecule_chembl_id": "CHEMBL_GOOD",
            "uniprot": "P_GOOD",
            "pchembl": 7.0,
            "assay_confidence_score": 9,
        },
        {
            **common,
            "molecule_chembl_id": "CHEMBL_WEAK",
            "uniprot": "P_WEAK",
            "pchembl": 4.0,
            "assay_confidence_score": 9,
        },
    ]).to_parquet(tmp_path / "human_activities.parquet")

    res = run_daina(
        tmp_path,
        chembl_fp,
        "--quality-policy",
        "high-confidence",
        "--scoring-method",
        "quality-hybrid",
    )

    assert res.returncode == 0, res.stderr
    scores = pd.read_csv(tmp_path / "daina.tsv", sep="\t")
    assert scores["target_id"].tolist() == ["P_GOOD"]
    assert scores.loc[0, "score"] <= scores.loc[0, "max_tanimoto"]
    assert scores.loc[0, "scoring_method"] == "quality-hybrid"


def test_claim_grade_policy_keeps_only_direct_dated_publication_evidence(
    tmp_path: Path,
) -> None:
    mutations = [
        ("ASSAY", {"assay_type": "F"}),
        ("CONF", {"assay_confidence_score": 8}),
        ("RELATIONSHIP", {"assay_relationship_type": "H"}),
        ("ENDPOINT", {"standard_type": "Potency"}),
        ("UNITS", {"standard_units": "uM"}),
        ("VALUE", {"standard_value": 0.0}),
        ("RELATION", {"standard_relation": ">"}),
        ("POTENCY", {"pchembl": 4.0}),
        ("DOCUMENT", {"document_type": "PATENT"}),
        ("UNDATED", {"document_year": None}),
        ("BINDINGDB", {"assay_source_name": "BINDINGDB"}),
        ("VARIANT", {"assay_variant_id": 42}),
        ("DUPLICATE", {"potential_duplicate": True}),
        ("INVALID", {"data_validity_comment": "Outside typical range"}),
    ]
    molecule_ids = ["CHEMBL_GOOD"] + [f"CHEMBL_{name}" for name, _ in mutations]
    chembl_fp = tmp_path / "fp_morgan2_2048.parquet"
    pd.DataFrame(
        [
            {
                "molecule_chembl_id": molecule_id,
                "bitvec": fingerprint_words("CCN"),
            }
            for molecule_id in molecule_ids
        ]
    ).to_parquet(chembl_fp)
    base = {
        "assay_type": "B",
        "target_type": "SINGLE PROTEIN",
        "assay_confidence_score": 9,
        "assay_relationship_type": "D",
        "standard_relation": "=",
        "standard_type": "Ki",
        "standard_value": 10.0,
        "standard_units": "nM",
        "pchembl": 8.0,
        "data_validity_comment": "Manually validated",
        "potential_duplicate": False,
        "document_type": "PUBLICATION",
        "document_year": 2022,
        "assay_source_name": "LITERATURE",
        "document_source_name": "LITERATURE",
        "assay_variant_id": None,
    }
    rows = [
        {
            **base,
            "molecule_chembl_id": "CHEMBL_GOOD",
            "uniprot": "P_GOOD",
        }
    ]
    for name, mutation in mutations:
        rows.append(
            {
                **base,
                **mutation,
                "molecule_chembl_id": f"CHEMBL_{name}",
                "uniprot": f"P_{name}",
            }
        )
    pd.DataFrame(rows).to_parquet(tmp_path / "human_activities.parquet")
    metadata = tmp_path / "daina.metadata.json"

    res = run_daina(
        tmp_path,
        chembl_fp,
        "--quality-policy",
        "claim-grade",
        "--out-metadata-json",
        str(metadata),
    )

    assert res.returncode == 0, res.stderr
    scores = pd.read_csv(tmp_path / "daina.tsv", sep="\t")
    assert scores["target_id"].tolist() == ["P_GOOD"]
    payload = json.loads(metadata.read_text())
    assert payload["quality_filter"]["before"] == len(rows)
    assert payload["quality_filter"]["after"] == 1
    assert payload["quality_filter"]["rejected_bindingdb_import"] == 1
    assert payload["quality_filter"]["rejected_variant"] == 1
