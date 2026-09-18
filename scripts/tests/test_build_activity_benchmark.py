from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
pytest.importorskip("rdkit")

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "eval" / "build_activity_benchmark.py"
SOURCE_TARGET_EXCLUSION_COLUMNS = [
    "schema_version",
    "source_db",
    "source_release",
    "source_document_key",
    "uniprot",
    "reason_code",
    "source_document_url",
    "source_document_target",
    "database_target",
    "reviewed_at_utc",
    "evidence_note",
]


def run_builder(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT)] + args,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _clusters(
    path: Path,
    accessions: list[str],
    exclusions: list[dict[str, object]] | None = None,
    cluster50_overrides: dict[str, str] | None = None,
    schema_version: str = "skinscout.target-cluster-map.v2",
    production_passes: bool = True,
    known_target_assistance: bool = False,
) -> Path:
    pd.DataFrame(
        [
            {
                "uniprot": accession,
                "target_cluster_30": f"c30_{accession}",
                "target_cluster_50": (cluster50_overrides or {}).get(
                    accession, f"c50_{accession}"
                ),
            }
            for accession in accessions
        ]
    ).to_csv(path, index=False)
    manifest = path.with_suffix(".manifest.json")
    payload: dict[str, object] = {
        "schema_version": schema_version,
        "parameters": {
            "engine": "MMseqs2 easy-cluster",
            "sequence_identity_thresholds": {"main": 0.30, "sensitivity": 0.50},
        },
        "inputs": {"sequence_universe_manifest": None},
        "artifact": {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "rows": len(accessions),
        },
    }
    if schema_version == "skinscout.screenable-target-cluster-map.v2":
        payload["production_contract"] = {"passes": production_passes}
        payload["universe_policy"] = {
            "evaluation_panel_used": False,
            "known_target_assistance": known_target_assistance,
            "union_target_count": len(accessions),
        }
    payload["excluded_targets"] = {
        "count": len(exclusions or []),
        "records": exclusions or [],
    }
    manifest.write_text(json.dumps(payload) + "\n")
    return path


def _cluster_manifest(path: Path) -> Path:
    return path.with_suffix(".manifest.json")


def _base_chembl(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "source_name": "ChEMBL",
        "source_release": "37",
        "source_license": "CC BY-SA 3.0",
        "activity_id": "ACT1",
        "assay_type": "B",
        "target_type": "SINGLE PROTEIN",
        "assay_confidence_score": 9,
        "assay_relationship_type": "D",
        "standard_relation": "=",
        "standard_type": "IC50",
        "standard_value": 100.0,
        "standard_units": "nM",
        "data_validity_comment": "",
        "potential_duplicate": False,
        "document_type": "PUBLICATION",
        "document_year": 2023,
        "document_chembl_id": "DOC1",
        "document_source_id": 1,
        "assay_variant_id": None,
        "assay_variant_accession": "",
        "assay_variant_mutation": "",
        "uniprot": "P11111",
        "molecule_chembl_id": "CHEMBL1",
        "smiles": "c1ccccc1O",
        "standard_inchi_key": "ISWSIDIOOBJBQZ-UHFFFAOYSA-N",
    }
    row.update(overrides)
    return row


def _base_bindingdb(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "source_db": "BindingDB",
        "source_origin": "BindingDB",
        "source_release": "2026-08",
        "source_license": "BindingDB terms",
        "chembl_derived_license_flag": False,
        "evidence_id": "BDB1",
        "affinity_type": "Ki",
        "relation": "=",
        "affinity_value": 1000.0,
        "affinity_unit": "nM",
        "source_doi": "10.1000/bdb1",
        "source_pmid": "",
        "source_patent": "",
        "source_article_id": "",
        "publication_date": "2024-06-01",
        "evidence_date_source": "publication",
        "single_chain_target": True,
        "uniprot": "P22222",
        "ligand_id": "BDBM1",
        "ligand_smiles": "c1ccncc1",
        "ligand_inchikey": "JUJWROOIHBZHMG-UHFFFAOYSA-N",
    }
    row.update(overrides)
    return row


def _base_gtopdb(**overrides: object) -> dict[str, object]:
    row = _base_bindingdb(
        source_db="GtoPdb",
        source_origin="GtoPdb expert-curated literature",
        source_release="2026.2",
        source_license="ODbL 1.0; contents CC BY-SA 4.0",
        evidence_id="GTP1",
        source_doi="10.1000/gtopdb",
        source_pmid="40000001",
        publication_date="2025-06-01",
        uniprot="P33333",
        ligand_id="GtoPdb:1",
        ligand_smiles="c1ccsc1",
        ligand_inchikey="TWDMSCGQKBTTDW-UHFFFAOYSA-N",
    )
    row.update(overrides)
    return row


def _base_source_target_exclusion(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "schema_version": "skinscout.activity-source-target-exclusions.v1",
        "source_db": "BindingDB",
        "source_release": "2026-08",
        "source_document_key": "patent:US20250257059",
        "uniprot": "Q92932",
        "reason_code": "source_document_target_mismatch",
        "source_document_url": (
            "https://patents.google.com/patent/US20250257059A1/en"
        ),
        "source_document_target": "PTPN2; PTPN1",
        "database_target": "PTPRN2 (UniProt Q92932)",
        "reviewed_at_utc": "2026-08-05T17:03:36Z",
        "evidence_note": "Primary patent target differs from the database target",
    }
    row.update(overrides)
    return row


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_manifest(
    path: Path,
    source: str,
    release: str,
    rows: int,
    artifact_name: str,
    *,
    schema_version: str | None = None,
) -> Path:
    manifest = path.with_suffix(".manifest.json")
    payload = {
        "source": {"name": source, "release": release, "license": "test"},
        "output_sha256": {artifact_name: _sha256(path)},
        "row_counts": {"activity_evidence": rows},
    }
    if schema_version is not None:
        payload["schema_version"] = schema_version
    manifest.write_text(json.dumps(payload) + "\n")
    return manifest


def _run_fixture(
    tmp_path: Path,
    chembl_rows: list[dict[str, object]] | None,
    bindingdb_rows: list[dict[str, object]] | None,
    clusters: list[str],
    extra: list[str] | None = None,
    manifests: bool = True,
    cluster_exclusions: list[dict[str, object]] | None = None,
    cluster50_overrides: dict[str, str] | None = None,
    source_target_exclusions: list[dict[str, object]] | None = None,
    gtopdb_rows: list[dict[str, object]] | None = None,
    cluster_schema_version: str = "skinscout.target-cluster-map.v2",
    cluster_production_passes: bool = True,
    cluster_known_target_assistance: bool = False,
) -> subprocess.CompletedProcess[str]:
    cluster_path = _clusters(
        tmp_path / "clusters.csv",
        clusters,
        exclusions=cluster_exclusions,
        cluster50_overrides=cluster50_overrides,
        schema_version=cluster_schema_version,
        production_passes=cluster_production_passes,
        known_target_assistance=cluster_known_target_assistance,
    )
    args: list[str] = [
        "--target-clusters",
        str(cluster_path),
        "--target-cluster-manifest",
        str(_cluster_manifest(cluster_path)),
    ]
    if source_target_exclusions is not None:
        exclusions_path = tmp_path / "source_target_exclusions.csv"
        pd.DataFrame(
            source_target_exclusions,
            columns=SOURCE_TARGET_EXCLUSION_COLUMNS,
        ).to_csv(exclusions_path, index=False)
        args.extend(["--source-target-exclusions", str(exclusions_path)])
    if chembl_rows is not None:
        chembl = tmp_path / "chembl.parquet"
        pd.DataFrame(chembl_rows).to_parquet(chembl, index=False)
        args.extend(["--chembl-evidence", str(chembl)])
        if manifests:
            chembl_release = str(chembl_rows[0].get("source_release", "37"))
            args.extend([
                "--chembl-manifest",
                str(_source_manifest(chembl, "ChEMBL", chembl_release, len(chembl_rows), "activity_evidence.parquet")),
            ])
    if bindingdb_rows is not None:
        bindingdb = tmp_path / "bindingdb.parquet"
        pd.DataFrame(bindingdb_rows).to_parquet(bindingdb, index=False)
        args.extend(["--bindingdb-evidence", str(bindingdb)])
        if manifests:
            bindingdb_release = str(bindingdb_rows[0].get("source_release", "2026-08"))
            args.extend([
                "--bindingdb-manifest",
                str(_source_manifest(bindingdb, "BindingDB", bindingdb_release, len(bindingdb_rows), "all.parquet")),
            ])
    if gtopdb_rows is not None:
        gtopdb = tmp_path / "gtopdb.parquet"
        pd.DataFrame(gtopdb_rows).to_parquet(gtopdb, index=False)
        args.extend(["--gtopdb-evidence", str(gtopdb)])
        if manifests:
            gtopdb_release = str(gtopdb_rows[0].get("source_release", "2026.2"))
            args.extend([
                "--gtopdb-manifest",
                str(
                    _source_manifest(
                        gtopdb,
                        "GtoPdb",
                        gtopdb_release,
                        len(gtopdb_rows),
                        "activity_evidence.parquet",
                        schema_version="gtopdb_activity_evidence.v1",
                    )
                ),
            ])
    args.extend(["--out-dir", str(tmp_path / "out")])
    if extra:
        args.extend(extra)
    return run_builder(args)


def test_claim_grade_filters_boundaries_labels_scaffolds_and_hashes(tmp_path: Path) -> None:
    chembl_rows = [
        _base_chembl(activity_id="train", document_chembl_id="DOC_TRAIN", document_year=2023),
        _base_chembl(activity_id="dev", document_chembl_id="DOC_DEV", document_year=2024, uniprot="P33333", molecule_chembl_id="CHEMBL3", standard_inchi_key="QUSNBJAOOMFDIB-UHFFFAOYSA-N", smiles="c1ccoc1", standard_value=100_000.0),
        _base_chembl(activity_id="test", document_chembl_id="DOC_TEST", document_year=2025, uniprot="P44444", molecule_chembl_id="CHEMBL4", standard_inchi_key="TWDMSCGQKBTTDW-UHFFFAOYSA-N", smiles="c1ccsc1", standard_type="Kd", data_validity_comment="Manually validated", standard_value=10.0),
        _base_chembl(activity_id="bad_assay", assay_type="F", document_chembl_id="DOC_BAD1"),
        _base_chembl(activity_id="bad_target", target_type="PROTEIN COMPLEX", document_chembl_id="DOC_BAD2"),
        _base_chembl(activity_id="bad_conf", assay_confidence_score=8, document_chembl_id="DOC_BAD3"),
        _base_chembl(activity_id="bad_reltype", assay_relationship_type="U", document_chembl_id="DOC_BAD4"),
        _base_chembl(activity_id="bad_relation", standard_relation="<", document_chembl_id="DOC_BAD5"),
        _base_chembl(activity_id="bad_unit", standard_units="uM", document_chembl_id="DOC_BAD6"),
        _base_chembl(activity_id="bad_value", standard_value=0, document_chembl_id="DOC_BAD7"),
        _base_chembl(activity_id="bad_endpoint", standard_type="Potency", document_chembl_id="DOC_BAD8"),
        _base_chembl(activity_id="bad_validity", data_validity_comment="Outside typical range", document_chembl_id="DOC_BAD9"),
        _base_chembl(activity_id="bad_dup", potential_duplicate=True, document_chembl_id="DOC_BAD10"),
        _base_chembl(activity_id="bad_doc_type", document_type="Patent", document_chembl_id="DOC_BAD11"),
        _base_chembl(activity_id="bad_date", document_year=None, document_chembl_id="DOC_BAD12"),
        _base_chembl(activity_id="bad_bindingdb_source", document_source_id=37, document_source_name="BINDINGDB", document_chembl_id="DOC_BAD13"),
        _base_chembl(activity_id="bad_variant", assay_variant_id=123, document_chembl_id="DOC_BAD14"),
        _base_chembl(activity_id="bad_blank_smiles", smiles="", document_chembl_id="DOC_BAD15"),
        _base_chembl(activity_id="bad_invalid_smiles", smiles="not-a-smiles", document_chembl_id="DOC_BAD16"),
        _base_chembl(activity_id="bad_publication_id", document_chembl_id=""),
    ]
    bindingdb_rows = [
        _base_bindingdb(
            ligand_smiles=(
                "OCCCC1CCN(CC1)c1ccc(Nc2ncc3ccc(=O)n(C4CC5CCC4C5)c3n2)cc1 "
                "|THB:23:24:27.28:30|"
            )
        ),
        _base_bindingdb(evidence_id="no_comment_negative", affinity_value=None, source_doi="10.1000/comment", uniprot="P55555", ligand_smiles="c1ccccc1Cl", ligand_inchikey="MVPPADPHJFYWMZ-UHFFFAOYSA-N"),
        _base_bindingdb(evidence_id="bad_bdb_endpoint", affinity_type="Potency", source_doi="10.1000/bad-endpoint"),
        _base_bindingdb(evidence_id="bad_bdb_relation", relation="<", source_doi="10.1000/bad-relation"),
        _base_bindingdb(evidence_id="bad_bdb_unit", affinity_unit="uM", source_doi="10.1000/bad-unit"),
        _base_bindingdb(evidence_id="bad_bdb_date_source", evidence_date_source="curation", source_doi="10.1000/bad-date-source"),
        _base_bindingdb(evidence_id="bad_bdb_single_chain", single_chain_target=False, source_doi="10.1000/bad-chain"),
        _base_bindingdb(evidence_id="bad_bdb_origin", source_origin="ChEMBL", chembl_derived_license_flag=True, source_doi="10.1000/bad-origin"),
        _base_bindingdb(evidence_id="bad_bdb_import", source_origin="PubChem", source_license="public domain", source_doi="10.1000/bad-import"),
        _base_bindingdb(evidence_id="bad_bdb_blank_smiles", ligand_smiles="", source_doi="10.1000/bad-smiles"),
        _base_bindingdb(evidence_id="bad_bdb_publication_id", source_doi="", source_pmid="", source_patent="", source_article_id=""),
    ]

    res = _run_fixture(tmp_path, chembl_rows, bindingdb_rows, ["P11111", "P22222", "P33333", "P44444", "P55555"])

    assert res.returncode == 0, res.stderr
    out = tmp_path / "out"
    train = pd.read_parquet(out / "train.parquet")
    dev = pd.read_parquet(out / "dev.parquet")
    test = pd.read_parquet(out / "test.parquet")
    assert train["evidence_id"].tolist() == ["train"]
    assert set(dev["evidence_id"]) == {"dev", "BDB1"}
    assert test["evidence_id"].tolist() == ["test"]
    assert train.loc[0, "pactivity"] == pytest.approx(7.0)
    assert train.loc[0, "activity_class"] == "strong_positive"
    low = dev.loc[dev["evidence_id"].eq("dev")].iloc[0]
    assert low["activity_class"] == "low_potency_quantitative"
    assert pd.isna(low["binary_label"])
    assert "no_comment_negative" not in set(pd.concat([train, dev, test])["evidence_id"])
    assert train.loc[0, "scaffold_smiles"] == "c1ccccc1"

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["splits"]["boundaries"] == {
        "dev": "2024-01-01..2024-12-31",
        "test": ">=2025-01-01",
        "train": "<=2023-12-31",
    }
    assert manifest["splits"]["counts"] == {"dev": 2, "test": 1, "train": 1}
    assert manifest["input_sha256"]["chembl_evidence"]
    assert manifest["input_sha256"]["bindingdb_evidence"]
    assert manifest["licenses"] == {"bindingdb": "test", "chembl": "test"}
    assert set(manifest["output_sha256"]) == {
        "train.parquet",
        "dev.parquet",
        "test.parquet",
        "temporal_test.parquet",
        "ligand_scaffold_cold.parquet",
        "target_cold30.parquet",
        "target_cold50.parquet",
        "dual_cold.parquet",
    }
    assert manifest["target_cluster_policy"]["main_threshold"] == "30%"
    assert manifest["target_cluster_policy"]["sensitivity_threshold"] == "50%"
    assert manifest["filters"]["ligand_structure_policy"]["unique_input_parse_modes"] == {
        "lenient_cx_metadata": 1,
        "strict": 3,
    }
    assert manifest["filters"]["ligand_structure_policy"][
        "invalid_nonblank_unique_inputs"
    ] == 1
    assert manifest["filters"]["ligand_structure_policy"][
        "invalid_nonblank_smiles_sha256"
    ] == [hashlib.sha256(b"not-a-smiles").hexdigest()]
    chembl_counts = manifest["filters"]["filter_counts"]["chembl"]
    for key in [
        "filtered_assay_type",
        "filtered_single_protein",
        "filtered_confidence_9",
        "filtered_relationship_d",
        "filtered_exact_relation",
        "filtered_nm_units",
        "filtered_positive_value",
        "filtered_endpoint",
        "filtered_validity",
        "filtered_nonduplicate",
        "filtered_document_publication",
        "filtered_dated",
        "filtered_exclude_chembl_bindingdb_source",
        "filtered_assay_variant",
        "filtered_nonblank_ligand_smiles",
        "filtered_publication_identifier",
    ]:
        assert chembl_counts[key] == 1
    assert chembl_counts["filtered_invalid_ligand_structure"] == 1
    bindingdb_counts = manifest["filters"]["filter_counts"]["bindingdb"]
    assert bindingdb_counts["filtered_positive_value"] == 1
    assert bindingdb_counts["filtered_endpoint"] == 1
    assert bindingdb_counts["filtered_exact_relation"] == 1
    assert bindingdb_counts["filtered_nm_units"] == 1
    assert bindingdb_counts["filtered_publication_date_evidence"] == 1
    assert bindingdb_counts["filtered_single_chain_target"] == 1
    assert bindingdb_counts["filtered_bindingdb_curated_origin"] == 2
    assert bindingdb_counts["filtered_nonblank_ligand_smiles"] == 1
    assert bindingdb_counts["filtered_publication_identifier"] == 1
    assert manifest["filters"]["chembl_claim_grade"]["bindingdb_source_ids_discovered"] == ["37"]
    assert manifest["evaluation_views"]["counts"]["temporal_test"] == 1
    assert (out / "temporal_test.parquet").exists()


def test_bindingdb_patent_identifier_is_canonicalized(tmp_path: Path) -> None:
    row = _base_bindingdb(
        source_doi="",
        source_pmid="",
        source_article_id="",
        source_patent="US 9,447,092 B2",
    )

    chembl_rows = [
        _base_chembl(
            activity_id="train",
            document_chembl_id="DOC_TRAIN",
            document_year=2023,
        ),
        _base_chembl(
            activity_id="test",
            document_chembl_id="DOC_TEST",
            document_year=2025,
            uniprot="P33333",
            molecule_chembl_id="CHEMBL3",
            smiles="c1ccoc1",
            standard_inchi_key="QUSNBJAOOMFDIB-UHFFFAOYSA-N",
        ),
    ]
    res = _run_fixture(
        tmp_path,
        chembl_rows,
        [row],
        ["P11111", "P22222", "P33333"],
    )

    assert res.returncode == 0, res.stderr
    dev = pd.read_parquet(tmp_path / "out" / "dev.parquet")
    assert dev.loc[0, "publication_key"] == "patent:US9447092B2"
    assert dev.loc[0, "source_document_id"] == "BindingDB:US 9,447,092 B2"


def test_source_target_exclusion_is_applied_and_provenance_bound(
    tmp_path: Path,
) -> None:
    chembl_rows = [
        _base_chembl(
            activity_id="train",
            document_chembl_id="DOC_TRAIN",
            document_year=2023,
        ),
        _base_chembl(
            activity_id="dev",
            document_chembl_id="DOC_DEV",
            document_year=2024,
            uniprot="P22222",
            molecule_chembl_id="CHEMBL2",
            smiles="c1ccncc1",
            standard_inchi_key="JUJWROOIHBZHMG-UHFFFAOYSA-N",
        ),
        _base_chembl(
            activity_id="test",
            document_chembl_id="DOC_TEST",
            document_year=2025,
            uniprot="P33333",
            molecule_chembl_id="CHEMBL3",
            smiles="c1ccoc1",
            standard_inchi_key="QUSNBJAOOMFDIB-UHFFFAOYSA-N",
        ),
    ]
    mismatch = _base_bindingdb(
        evidence_id="wrong-patent-target",
        source_doi="",
        source_pmid="",
        source_article_id="",
        source_patent="US20250257059",
        publication_date="2025-06-01",
        uniprot="Q92932",
        ligand_id="BDB-WRONG",
        ligand_smiles="c1ccccc1Cl",
        ligand_inchikey="MVPPADPHJFYWMZ-UHFFFAOYSA-N",
    )

    res = _run_fixture(
        tmp_path,
        chembl_rows,
        [mismatch],
        ["P11111", "P22222", "P33333", "Q92932"],
        source_target_exclusions=[_base_source_target_exclusion()],
    )

    assert res.returncode == 0, res.stderr
    observed = pd.concat(
        [
            pd.read_parquet(tmp_path / "out" / f"{split}.parquet")
            for split in ("train", "dev", "test")
        ]
    )
    assert "wrong-patent-target" not in set(observed["evidence_id"])
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    exclusion_path = tmp_path / "source_target_exclusions.csv"
    assert manifest["input_sha256"]["source_target_exclusions"] == _sha256(
        exclusion_path
    )
    audit = manifest["filters"]["source_target_exclusions"]
    assert audit["applied"] is True
    assert audit["all_rules_matched"] is True
    assert audit["rule_count"] == 1
    assert audit["total_filtered_rows"] == 1
    assert audit["filtered_rows_by_source"] == {"bindingdb": 1}
    assert audit["artifact"] == {
        "path": str(exclusion_path.resolve()),
        "rows": 1,
        "sha256": _sha256(exclusion_path),
    }
    assert audit["rules"][0]["matched_rows"] == 1
    bindingdb_counts = manifest["filters"]["filter_counts"]["bindingdb"]
    assert bindingdb_counts["accepted_before_source_target_exclusions"] == 1
    assert bindingdb_counts["filtered_source_document_target_mismatch"] == 1
    assert bindingdb_counts["accepted"] == 0


def test_stale_source_target_exclusion_rule_fails_closed_and_removes_outputs(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "manifest.json").write_text("stale\n")
    chembl_rows = [
        _base_chembl(document_year=2023, document_chembl_id="DOC_TRAIN"),
        _base_chembl(
            activity_id="dev",
            document_year=2024,
            document_chembl_id="DOC_DEV",
            uniprot="P22222",
            molecule_chembl_id="CHEMBL2",
            smiles="c1ccncc1",
            standard_inchi_key="JUJWROOIHBZHMG-UHFFFAOYSA-N",
        ),
        _base_chembl(
            activity_id="test",
            document_year=2025,
            document_chembl_id="DOC_TEST",
            uniprot="P33333",
            molecule_chembl_id="CHEMBL3",
            smiles="c1ccoc1",
            standard_inchi_key="QUSNBJAOOMFDIB-UHFFFAOYSA-N",
        ),
    ]

    res = _run_fixture(
        tmp_path,
        chembl_rows,
        None,
        ["P11111", "P22222", "P33333"],
        source_target_exclusions=[_base_source_target_exclusion()],
    )

    assert res.returncode != 0
    assert "did not match any normalized claim-grade row" in res.stderr
    assert not (out_dir / "manifest.json").exists()


def test_non_normalized_source_target_exclusion_key_is_rejected(
    tmp_path: Path,
) -> None:
    exclusions_path = tmp_path / "bad_exclusions.csv"
    pd.DataFrame(
        [
            _base_source_target_exclusion(
                source_document_key="patent:US 20250257059"
            )
        ],
        columns=SOURCE_TARGET_EXCLUSION_COLUMNS,
    ).to_csv(exclusions_path, index=False)
    clusters = _clusters(tmp_path / "clusters.csv", ["P11111"])
    chembl = tmp_path / "chembl.parquet"
    pd.DataFrame([_base_chembl()]).to_parquet(chembl, index=False)

    res = run_builder(
        [
            "--chembl-evidence",
            str(chembl),
            "--chembl-manifest",
            str(_source_manifest(chembl, "ChEMBL", "37", 1, "activity_evidence.parquet")),
            "--target-clusters",
            str(clusters),
            "--target-cluster-manifest",
            str(_cluster_manifest(clusters)),
            "--source-target-exclusions",
            str(exclusions_path),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )

    assert res.returncode != 0
    assert "source_document_key is not normalized" in res.stderr


def test_missing_target_cluster_assignment_fails_closed_and_removes_outputs(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    for name in ["train.parquet", "dev.parquet", "test.parquet", "manifest.json"]:
        (out_dir / name).write_text("stale\n")
    chembl = tmp_path / "chembl.parquet"
    pd.DataFrame([_base_chembl()]).to_parquet(chembl, index=False)
    chembl_manifest = _source_manifest(chembl, "ChEMBL", "37", 1, "activity_evidence.parquet")
    clusters = _clusters(tmp_path / "clusters.csv", ["P99999"])

    res = run_builder([
        "--chembl-evidence",
        str(chembl),
        "--chembl-manifest",
        str(chembl_manifest),
        "--target-clusters",
        str(clusters),
        "--target-cluster-manifest",
        str(_cluster_manifest(clusters)),
        "--out-dir",
        str(out_dir),
    ])

    assert res.returncode != 0
    assert "Missing target cluster assignment for ChEMBL target P11111" in res.stderr
    assert not any((out_dir / name).exists() for name in ["train.parquet", "dev.parquet", "test.parquet", "manifest.json"])


def test_screenable_target_cluster_contract_is_accepted_for_benchmark_leakage(
    tmp_path: Path,
) -> None:
    rows = [
        _base_chembl(),
        _base_chembl(
            activity_id="ACT2",
            document_chembl_id="DOC2",
            document_year=2024,
            uniprot="P22222",
            molecule_chembl_id="CHEMBL2",
            smiles="c1ccncc1",
            standard_inchi_key="JUJWROOIHBZHMG-UHFFFAOYSA-N",
        ),
        _base_chembl(
            activity_id="ACT3",
            document_chembl_id="DOC3",
            document_year=2025,
            uniprot="P33333",
            molecule_chembl_id="CHEMBL3",
            smiles="c1ccoc1",
            standard_inchi_key="QUSNBJAOOMFDIB-UHFFFAOYSA-N",
        ),
    ]
    res = _run_fixture(
        tmp_path,
        rows,
        None,
        ["P11111", "P22222", "P33333"],
        cluster_schema_version="skinscout.screenable-target-cluster-map.v2",
    )

    assert res.returncode == 0, res.stderr
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    cluster_input = manifest["target_cluster_policy"]["required_input"]
    assert cluster_input["manifest_schema_version"] == (
        "skinscout.screenable-target-cluster-map.v2"
    )
    assert cluster_input["production_contract"]["passes"] is True
    assert cluster_input["universe_policy"]["known_target_assistance"] is False


@pytest.mark.parametrize(
    ("production_passes", "known_target_assistance", "message"),
    [
        (False, False, "production contract must pass"),
        (True, True, "must not use known-target assistance"),
    ],
)
def test_screenable_target_cluster_contract_fails_closed(
    tmp_path: Path,
    production_passes: bool,
    known_target_assistance: bool,
    message: str,
) -> None:
    res = _run_fixture(
        tmp_path,
        [_base_chembl()],
        None,
        ["P11111"],
        cluster_schema_version="skinscout.screenable-target-cluster-map.v2",
        cluster_production_passes=production_passes,
        cluster_known_target_assistance=known_target_assistance,
    )

    assert res.returncode != 0
    assert message in res.stderr


def test_screenable_map_audited_exclusion_overrides_general_cluster_assignment(
    tmp_path: Path,
) -> None:
    rows = [
        _base_chembl(),
        _base_chembl(
            activity_id="EXCLUDED",
            document_chembl_id="DOC_EXCLUDED",
            uniprot="PEXCLUDED",
            molecule_chembl_id="CHEMBLEXCLUDED",
            smiles="CCO",
            standard_inchi_key="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        ),
        _base_chembl(
            activity_id="DEV",
            document_chembl_id="DOC_DEV",
            document_year=2024,
            uniprot="P22222",
            molecule_chembl_id="CHEMBLDEV",
            smiles="c1ccncc1",
            standard_inchi_key="JUJWROOIHBZHMG-UHFFFAOYSA-N",
        ),
        _base_chembl(
            activity_id="TEST",
            document_chembl_id="DOC_TEST",
            document_year=2025,
            uniprot="P33333",
            molecule_chembl_id="CHEMBLTEST",
            smiles="c1ccoc1",
            standard_inchi_key="QUSNBJAOOMFDIB-UHFFFAOYSA-N",
        ),
    ]
    exclusion = {
        "uniprot": "PEXCLUDED",
        "reason": "unresolved_or_obsolete_uniprot_record",
        "uniprot_release": "2026_02",
    }

    res = _run_fixture(
        tmp_path,
        rows,
        None,
        ["P11111", "P22222", "P33333", "PEXCLUDED"],
        cluster_exclusions=[exclusion],
        cluster_schema_version="skinscout.screenable-target-cluster-map.v2",
    )

    assert res.returncode == 0, res.stderr
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["filters"]["filter_counts"]["chembl"][
        "filtered_target_sequence_exclusion"
    ] == 1


def test_audited_sequence_exclusion_is_counted_and_only_declared_target_is_removed(
    tmp_path: Path,
) -> None:
    rows = [
        _base_chembl(activity_id="train", document_chembl_id="DOC1", document_year=2023),
        _base_chembl(
            activity_id="excluded",
            document_chembl_id="DOC_EXCLUDED",
            document_year=2023,
            uniprot="PEXCLUDED",
            molecule_chembl_id="CHEMBLEXCLUDED",
            standard_inchi_key="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
            smiles="CCO",
        ),
        _base_chembl(
            activity_id="dev",
            document_chembl_id="DOC2",
            document_year=2024,
            uniprot="P22222",
            molecule_chembl_id="CHEMBL2",
            standard_inchi_key="JUJWROOIHBZHMG-UHFFFAOYSA-N",
            smiles="c1ccncc1",
        ),
        _base_chembl(
            activity_id="test",
            document_chembl_id="DOC3",
            document_year=2025,
            uniprot="P33333",
            molecule_chembl_id="CHEMBL3",
            standard_inchi_key="QUSNBJAOOMFDIB-UHFFFAOYSA-N",
            smiles="c1ccoc1",
        ),
    ]
    exclusion = {
        "uniprot": "PEXCLUDED",
        "reason": "taxonomy_mismatch",
        "organism_id": "9601",
        "uniprot_release": "2026_02",
    }

    res = _run_fixture(
        tmp_path,
        rows,
        None,
        ["P11111", "P22222", "P33333"],
        cluster_exclusions=[exclusion],
    )

    assert res.returncode == 0, res.stderr
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["filters"]["filter_counts"]["chembl"][
        "filtered_target_sequence_exclusion"
    ] == 1
    assert manifest["target_cluster_policy"]["audited_sequence_exclusions"] == {
        "count": 1,
        "records": [exclusion],
        "filtered_evidence_rows_by_source": {"chembl": 1},
        "policy": (
            "Only exclusions carried by the provenance-bound target cluster manifest are "
            "removed; every other claim-grade evidence target requires a sequence cluster "
            "assignment."
        ),
    }
    observed = set()
    for split in ("train", "dev", "test"):
        observed.update(pd.read_parquet(tmp_path / "out" / f"{split}.parquet")["uniprot"])
    assert "PEXCLUDED" not in observed


def test_repeated_pair_is_kept_only_in_its_first_observation_split(tmp_path: Path) -> None:
    rows = [
        _base_chembl(activity_id="train", document_chembl_id="DOC1", document_year=2023),
        _base_chembl(activity_id="dev", document_chembl_id="DOC2", document_year=2024, standard_value=50.0),
        _base_chembl(
            activity_id="dev_novel",
            document_chembl_id="DOC3",
            document_year=2024,
            uniprot="P22222",
            molecule_chembl_id="CHEMBL2",
            smiles="c1ccncc1",
            standard_inchi_key="JUJWROOIHBZHMG-UHFFFAOYSA-N",
        ),
        _base_chembl(
            activity_id="test_novel",
            document_chembl_id="DOC4",
            document_year=2025,
            uniprot="P33333",
            molecule_chembl_id="CHEMBL3",
            smiles="c1ccoc1",
            standard_inchi_key="QUSNBJAOOMFDIB-UHFFFAOYSA-N",
        ),
    ]

    res = _run_fixture(tmp_path, rows, None, ["P11111", "P22222", "P33333"])

    assert res.returncode == 0, res.stderr
    out = tmp_path / "out"
    assert pd.read_parquet(out / "dev.parquet")["evidence_id"].tolist() == ["dev_novel"]
    manifest = json.loads((out / "manifest.json").read_text())
    first_observation = manifest["splits"]["first_observation_filter"]
    assert first_observation["removed_total"] == 1
    assert first_observation["removed_pair_seen_in_prior_split"] == 1
    assert first_observation["removed_publication_seen_in_prior_split"] == 0
    assert manifest["raw_temporal_overlap_audit"]["pair"]["train_dev"]["count"] == 1
    assert manifest["leakage_overlap_audit"]["pair"]["train_dev"]["count"] == 0


@pytest.mark.parametrize("later_inchikey", ["", "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"])
def test_structure_identity_survives_missing_keys_and_source_ids(tmp_path: Path, later_inchikey: str) -> None:
    rows = [
        _base_chembl(activity_id="train", smiles="CCO", standard_inchi_key="",
                     molecule_chembl_id="CHEMBL_A", document_year=2023),
        _base_chembl(activity_id="test", smiles="OCC", standard_inchi_key=later_inchikey,
                     molecule_chembl_id="CHEMBL_B", document_year=2025, document_chembl_id="DOC_TEST"),
        _base_chembl(activity_id="dev_novel", document_year=2024, document_chembl_id="DOC_DEV",
                     uniprot="P22222", smiles="c1ccncc1"),
        _base_chembl(activity_id="test_novel", document_year=2025, document_chembl_id="DOC_NOVEL",
                     uniprot="P33333", smiles="c1ccoc1"),
    ]
    result = _run_fixture(tmp_path, rows, None, ["P11111", "P22222", "P33333"])
    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / "out/manifest.json").read_text())
    assert manifest["raw_temporal_overlap_audit"]["pair"]["train_test"]["count"] == 1
    assert manifest["splits"]["first_observation_filter"]["removed_pair_seen_in_prior_split"] == 1
    assert pd.read_parquet(tmp_path / "out/test.parquet")["evidence_id"].tolist() == ["test_novel"]


def test_opposite_defined_stereochemistry_remains_distinct_across_splits(tmp_path: Path) -> None:
    rows = [
        _base_chembl(activity_id="train", smiles="C[C@H](O)F", standard_inchi_key="",
                     molecule_chembl_id="CHEMBL_A", document_year=2023),
        _base_chembl(activity_id="test_stereoisomer", smiles="C[C@@H](O)F", standard_inchi_key="",
                     molecule_chembl_id="CHEMBL_B", document_year=2025, document_chembl_id="DOC_TEST"),
        _base_chembl(activity_id="dev_novel", document_year=2024, document_chembl_id="DOC_DEV",
                     uniprot="P22222", smiles="c1ccncc1"),
    ]
    result = _run_fixture(tmp_path, rows, None, ["P11111", "P22222"])
    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / "out/manifest.json").read_text())
    assert manifest["raw_temporal_overlap_audit"]["pair"]["train_test"]["count"] == 0
    assert manifest["splits"]["first_observation_filter"]["removed_pair_seen_in_prior_split"] == 0
    assert pd.read_parquet(tmp_path / "out/test.parquet")["evidence_id"].tolist() == ["test_stereoisomer"]
    import rdkit

    assert manifest["versions"]["rdkit"] == rdkit.__version__


def test_pair_source_document_dedupe_and_manifest_hashes(tmp_path: Path) -> None:
    rows = [
        _base_chembl(activity_id="one", document_chembl_id="DOC1", document_year=2023),
        _base_chembl(activity_id="two", document_chembl_id="DOC1", document_year=2023, standard_value=50.0),
        _base_chembl(
            activity_id="dev",
            document_chembl_id="DOC2",
            document_year=2024,
            uniprot="P22222",
            molecule_chembl_id="CHEMBL2",
            smiles="c1ccncc1",
            standard_inchi_key="JUJWROOIHBZHMG-UHFFFAOYSA-N",
        ),
        _base_chembl(
            activity_id="test",
            document_chembl_id="DOC3",
            document_year=2025,
            uniprot="P33333",
            molecule_chembl_id="CHEMBL3",
            smiles="c1ccoc1",
            standard_inchi_key="QUSNBJAOOMFDIB-UHFFFAOYSA-N",
        ),
    ]

    res = _run_fixture(tmp_path, rows, None, ["P11111", "P22222", "P33333"])

    assert res.returncode == 0, res.stderr
    train = pd.read_parquet(tmp_path / "out" / "train.parquet")
    assert train["evidence_id"].tolist() == ["one"]
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["filters"]["pair_source_document_duplicates_removed"] == 1
    assert manifest["output_sha256"]["train.parquet"]


def test_temporal_split_allows_scaffold_and_target_overlap_and_derives_cold_views(tmp_path: Path) -> None:
    rows = [
        _base_chembl(activity_id="train", document_chembl_id="DOC_TRAIN", document_year=2023),
        _base_chembl(
            activity_id="dev",
            document_chembl_id="DOC_DEV",
            document_year=2024,
            uniprot="P22222",
            molecule_chembl_id="CHEMBL2",
            smiles="c1ccncc1",
            standard_inchi_key="JUJWROOIHBZHMG-UHFFFAOYSA-N",
        ),
        _base_chembl(
            activity_id="test_overlap_allowed",
            document_chembl_id="DOC_TEST_OVERLAP",
            document_year=2025,
            molecule_chembl_id="CHEMBL4",
            smiles="c1ccccc1Cl",
            standard_inchi_key="MVPPADPHJFYWMZ-UHFFFAOYSA-N",
        ),
        _base_chembl(
            activity_id="test_cold",
            document_chembl_id="DOC_TEST_COLD",
            document_year=2025,
            uniprot="P33333",
            molecule_chembl_id="CHEMBL3",
            smiles="c1ccoc1",
            standard_inchi_key="QUSNBJAOOMFDIB-UHFFFAOYSA-N",
        ),
    ]

    res = _run_fixture(tmp_path, rows, None, ["P11111", "P22222", "P33333"])

    assert res.returncode == 0, res.stderr
    out = tmp_path / "out"
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["leakage_overlap_audit"]["scaffold"]["train_test"]["count"] == 1
    assert manifest["leakage_overlap_audit"]["target_cluster_30"]["train_test"]["count"] == 1
    assert manifest["leakage_overlap_audit"]["pair"]["train_test"]["count"] == 0
    assert manifest["evaluation_views"]["counts"] == {
        "dual_cold": 1,
        "ligand_scaffold_cold": 1,
        "target_cold30": 1,
        "target_cold50": 1,
        "temporal_test": 2,
    }
    temporal = pd.read_parquet(out / "temporal_test.parquet")
    assert set(temporal["evaluation_view"]) == {"temporal_test"}
    cold = pd.read_parquet(out / "dual_cold.parquet")
    assert cold["evidence_id"].tolist() == ["test_cold"]
    assert cold.loc[0, "absent_scaffold_from_prior_splits"]
    assert cold.loc[0, "absent_target_cluster_30_from_prior_splits"]


def test_dual_cold_requires_both_30_and_50_percent_target_coldness(
    tmp_path: Path,
) -> None:
    rows = [
        _base_chembl(
            activity_id="train",
            document_chembl_id="DOC_TRAIN",
            document_year=2023,
            uniprot="P11111",
        ),
        _base_chembl(
            activity_id="test",
            document_chembl_id="DOC_TEST",
            document_year=2025,
            uniprot="P22222",
            molecule_chembl_id="CHEMBL2",
            smiles="c1ccncc1",
            standard_inchi_key="JUJWROOIHBZHMG-UHFFFAOYSA-N",
        ),
        _base_chembl(
            activity_id="dev",
            document_chembl_id="DOC_DEV",
            document_year=2024,
            uniprot="P33333",
            molecule_chembl_id="CHEMBL3",
            smiles="c1ccoc1",
            standard_inchi_key="QUSNBJAOOMFDIB-UHFFFAOYSA-N",
        ),
    ]

    result = _run_fixture(
        tmp_path,
        rows,
        None,
        ["P11111", "P22222", "P33333"],
        cluster50_overrides={"P11111": "shared50", "P22222": "shared50"},
    )

    assert result.returncode == 0, result.stderr
    out = tmp_path / "out"
    assert len(pd.read_parquet(out / "target_cold30.parquet")) == 1
    assert len(pd.read_parquet(out / "target_cold50.parquet")) == 0
    assert len(pd.read_parquet(out / "dual_cold.parquet")) == 0


def test_cross_source_doi_publication_is_kept_only_in_first_observation_split(tmp_path: Path) -> None:
    chembl_rows = [
        _base_chembl(
            activity_id="train",
            document_chembl_id="CHEMBL_DOC",
            document_year=2023,
            source_doi="https://doi.org/10.5555/SAME",
        ),
        _base_chembl(
            activity_id="dev",
            document_chembl_id="DEV_DOC",
            document_year=2024,
            source_doi="10.5555/dev",
            uniprot="P22222",
            molecule_chembl_id="CHEMBL2",
            smiles="c1ccncc1",
            standard_inchi_key="JUJWROOIHBZHMG-UHFFFAOYSA-N",
        ),
    ]
    bindingdb_rows = [
        _base_bindingdb(
            evidence_id="test_same_publication",
            publication_date="2025-02-01",
            source_doi="doi:10.5555/same",
            uniprot="P33333",
            ligand_smiles="c1ccoc1",
            ligand_inchikey="QUSNBJAOOMFDIB-UHFFFAOYSA-N",
        ),
        _base_bindingdb(
            evidence_id="test_novel_publication",
            publication_date="2025-03-01",
            source_doi="10.5555/novel-test",
            uniprot="P44444",
            ligand_smiles="c1ccsc1",
            ligand_inchikey="TWDMSCGQKBTTDW-UHFFFAOYSA-N",
        ),
    ]

    res = _run_fixture(tmp_path, chembl_rows, bindingdb_rows, ["P11111", "P22222", "P33333", "P44444"])

    assert res.returncode == 0, res.stderr
    out = tmp_path / "out"
    assert pd.read_parquet(out / "test.parquet")["evidence_id"].tolist() == ["test_novel_publication"]
    manifest = json.loads((out / "manifest.json").read_text())
    first_observation = manifest["splits"]["first_observation_filter"]
    assert first_observation["removed_total"] == 1
    assert first_observation["removed_publication_seen_in_prior_split"] == 1
    assert manifest["raw_temporal_overlap_audit"]["publication"]["train_test"]["count"] == 1
    assert manifest["leakage_overlap_audit"]["publication"]["train_test"]["count"] == 0


def test_bindingdb_required_release_is_cli_bound_not_hardcoded(tmp_path: Path) -> None:
    chembl_rows = [
        _base_chembl(activity_id="train", document_chembl_id="DOC1", document_year=2023),
        _base_chembl(activity_id="dev", document_chembl_id="DOC2", document_year=2024, uniprot="P22222", molecule_chembl_id="CHEMBL2", smiles="c1ccncc1", standard_inchi_key="JUJWROOIHBZHMG-UHFFFAOYSA-N"),
    ]
    bindingdb_rows = [
        _base_bindingdb(
            evidence_id="future_bound_release",
            source_release="2026-09",
            publication_date="2025-03-01",
            uniprot="P33333",
            ligand_smiles="c1ccoc1",
            ligand_inchikey="QUSNBJAOOMFDIB-UHFFFAOYSA-N",
            source_doi="10.1000/future-bound",
        )
    ]

    res = _run_fixture(
        tmp_path,
        chembl_rows,
        bindingdb_rows,
        ["P11111", "P22222", "P33333"],
        extra=["--bindingdb-required-release", "2026-09"],
    )

    assert res.returncode == 0, res.stderr
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["filters"]["bindingdb_claim_grade"]["required_release"] == "2026-09"
    assert manifest["filters"]["bindingdb_claim_grade"]["default_required_release"] == "2026-08"


def test_gtopdb_is_validated_as_a_third_claim_grade_source(tmp_path: Path) -> None:
    chembl_rows = [
        _base_chembl(activity_id="train", document_chembl_id="DOC1", document_year=2023),
        _base_chembl(
            activity_id="dev",
            document_chembl_id="DOC2",
            document_year=2024,
            uniprot="P22222",
            molecule_chembl_id="CHEMBL2",
            smiles="c1ccncc1",
            standard_inchi_key="JUJWROOIHBZHMG-UHFFFAOYSA-N",
        ),
    ]

    res = _run_fixture(
        tmp_path,
        chembl_rows,
        None,
        ["P11111", "P22222", "P33333"],
        gtopdb_rows=[_base_gtopdb()],
    )

    assert res.returncode == 0, res.stderr
    test = pd.read_parquet(tmp_path / "out" / "test.parquet")
    assert test["source_db"].tolist() == ["GtoPdb"]
    assert test["evidence_id"].tolist() == ["GTP1"]
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["source_manifests"]["gtopdb"]["validated"][
        "required_release"
    ] == "2026.2"
    assert manifest["filters"]["gtopdb_claim_grade"]["source_manifest_schema"] == (
        "gtopdb_activity_evidence.v1"
    )
    assert manifest["filters"]["filter_counts"]["gtopdb"]["accepted"] == 1


def test_gtopdb_manifest_schema_fails_closed(tmp_path: Path) -> None:
    cluster_path = _clusters(
        tmp_path / "clusters.csv",
        ["P11111", "P22222", "P33333"],
    )
    gtopdb = tmp_path / "gtopdb.parquet"
    pd.DataFrame([_base_gtopdb()]).to_parquet(gtopdb, index=False)
    manifest = _source_manifest(
        gtopdb,
        "GtoPdb",
        "2026.2",
        1,
        "activity_evidence.parquet",
        schema_version="gtopdb_activity_evidence.v0",
    )

    res = run_builder(
        [
            "--gtopdb-evidence",
            str(gtopdb),
            "--gtopdb-manifest",
            str(manifest),
            "--target-clusters",
            str(cluster_path),
            "--target-cluster-manifest",
            str(_cluster_manifest(cluster_path)),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )

    assert res.returncode != 0
    assert "source manifest schema must be gtopdb_activity_evidence.v1" in res.stderr


def test_required_and_mismatched_source_manifests_fail_closed(tmp_path: Path) -> None:
    res = _run_fixture(
        tmp_path,
        [
            _base_chembl(activity_id="train", document_chembl_id="DOC1", document_year=2023),
            _base_chembl(activity_id="dev", document_chembl_id="DOC2", document_year=2024, uniprot="P22222", molecule_chembl_id="CHEMBL2", smiles="c1ccncc1", standard_inchi_key="JUJWROOIHBZHMG-UHFFFAOYSA-N"),
            _base_chembl(activity_id="test", document_chembl_id="DOC3", document_year=2025, uniprot="P33333", molecule_chembl_id="CHEMBL3", smiles="c1ccoc1", standard_inchi_key="QUSNBJAOOMFDIB-UHFFFAOYSA-N"),
        ],
        None,
        ["P11111", "P22222", "P33333"],
        manifests=False,
    )
    assert res.returncode != 0
    assert "--chembl-manifest is required" in res.stderr

    chembl = tmp_path / "chembl_bad_manifest.parquet"
    rows = [
        _base_chembl(activity_id="train", document_chembl_id="DOC1", document_year=2023),
        _base_chembl(activity_id="dev", document_chembl_id="DOC2", document_year=2024, uniprot="P22222", molecule_chembl_id="CHEMBL2", smiles="c1ccncc1", standard_inchi_key="JUJWROOIHBZHMG-UHFFFAOYSA-N"),
        _base_chembl(activity_id="test", document_chembl_id="DOC3", document_year=2025, uniprot="P33333", molecule_chembl_id="CHEMBL3", smiles="c1ccoc1", standard_inchi_key="QUSNBJAOOMFDIB-UHFFFAOYSA-N"),
    ]
    pd.DataFrame(rows).to_parquet(chembl, index=False)
    bad_manifest = chembl.with_suffix(".manifest.json")
    bad_manifest.write_text(
        json.dumps(
            {
                "source": {"name": "ChEMBL", "release": "36", "license": "test"},
                "output_sha256": {"activity_evidence.parquet": "bad"},
                "row_counts": {"activity_evidence": len(rows)},
            }
        )
        + "\n"
    )

    cluster_path = _clusters(
        tmp_path / "clusters_bad_manifest.csv",
        ["P11111", "P22222", "P33333"],
    )
    res = run_builder([
        "--chembl-evidence",
        str(chembl),
        "--chembl-manifest",
        str(bad_manifest),
        "--target-clusters",
        str(cluster_path),
        "--target-cluster-manifest",
        str(_cluster_manifest(cluster_path)),
        "--out-dir",
        str(tmp_path / "out_bad_manifest"),
    ])

    assert res.returncode != 0
    assert "release must be 37" in res.stderr


def test_source_manifest_without_license_fails_closed(tmp_path: Path) -> None:
    chembl = tmp_path / "chembl_unlicensed.parquet"
    rows = [_base_chembl()]
    pd.DataFrame(rows).to_parquet(chembl, index=False)
    manifest = _source_manifest(
        chembl,
        "ChEMBL",
        "37",
        len(rows),
        "activity_evidence.parquet",
    )
    payload = json.loads(manifest.read_text())
    payload["source"]["license"] = ""
    manifest.write_text(json.dumps(payload) + "\n")
    cluster_path = _clusters(tmp_path / "clusters.csv", ["P11111"])

    res = run_builder(
        [
            "--chembl-evidence",
            str(chembl),
            "--chembl-manifest",
            str(manifest),
            "--target-clusters",
            str(cluster_path),
            "--target-cluster-manifest",
            str(_cluster_manifest(cluster_path)),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )

    assert res.returncode != 0
    assert "source manifest must declare a nonblank license" in res.stderr


def test_assay_variant_minus_one_is_rejected(tmp_path: Path) -> None:
    rows = [
        _base_chembl(activity_id="train", document_chembl_id="DOC1", document_year=2023),
        _base_chembl(activity_id="variant_minus_one", document_chembl_id="DOC_BAD", document_year=2023, assay_variant_id=-1),
        _base_chembl(activity_id="dev", document_chembl_id="DOC2", document_year=2024, uniprot="P22222", molecule_chembl_id="CHEMBL2", smiles="c1ccncc1", standard_inchi_key="JUJWROOIHBZHMG-UHFFFAOYSA-N"),
        _base_chembl(activity_id="test", document_chembl_id="DOC3", document_year=2025, uniprot="P33333", molecule_chembl_id="CHEMBL3", smiles="c1ccoc1", standard_inchi_key="QUSNBJAOOMFDIB-UHFFFAOYSA-N"),
    ]

    res = _run_fixture(tmp_path, rows, None, ["P11111", "P22222", "P33333"])

    assert res.returncode == 0, res.stderr
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["filters"]["filter_counts"]["chembl"]["filtered_assay_variant"] == 1
    assert "variant_minus_one" not in set(pd.read_parquet(tmp_path / "out" / "train.parquet")["evidence_id"])


def test_pre_read_max_row_gate_rejects_before_manifest_requirement(tmp_path: Path) -> None:
    chembl = tmp_path / "chembl_too_many.parquet"
    pd.DataFrame([
        _base_chembl(activity_id="one"),
        _base_chembl(activity_id="two", document_chembl_id="DOC2"),
    ]).to_parquet(chembl, index=False)

    cluster_path = _clusters(tmp_path / "clusters_too_many.csv", ["P11111"])
    res = run_builder([
        "--chembl-evidence",
        str(chembl),
        "--target-clusters",
        str(cluster_path),
        "--target-cluster-manifest",
        str(_cluster_manifest(cluster_path)),
        "--out-dir",
        str(tmp_path / "out_too_many"),
        "--max-input-rows",
        "1",
    ])

    assert res.returncode != 0
    assert "exceed fail-closed --max-input-rows=1" in res.stderr
    assert "--chembl-manifest is required" not in res.stderr
