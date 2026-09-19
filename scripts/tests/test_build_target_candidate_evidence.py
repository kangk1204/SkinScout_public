from __future__ import annotations

import importlib.util
import json
import sqlite3
import subprocess
from pathlib import Path

import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build_target_candidate_evidence.py"
PYTHON = Path(__import__("sys").executable)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "build_target_candidate_evidence", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _create_sqlite(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE assays (
          assay_id INTEGER PRIMARY KEY, description TEXT, assay_organism TEXT,
          assay_tax_id INTEGER, assay_strain TEXT, assay_tissue TEXT,
          assay_cell_type TEXT, assay_subcellular_fraction TEXT, cell_id INTEGER,
          bao_format TEXT
        );
        CREATE TABLE cell_dictionary (
          cell_id INTEGER PRIMARY KEY, cell_name TEXT, cell_description TEXT,
          cell_source_tissue TEXT, cell_source_organism TEXT,
          cell_source_tax_id INTEGER, cellosaurus_id TEXT
        );
        CREATE TABLE molecule_dictionary (
          molregno INTEGER PRIMARY KEY, chembl_id TEXT, pref_name TEXT
        );
        CREATE TABLE molecule_hierarchy (
          molregno INTEGER PRIMARY KEY, parent_molregno INTEGER
        );
        CREATE TABLE compound_structures (
          molregno INTEGER PRIMARY KEY, canonical_smiles TEXT,
          standard_inchi_key TEXT
        );
        CREATE TABLE drug_mechanism (
          mec_id INTEGER PRIMARY KEY, molregno INTEGER, tid INTEGER,
          mechanism_of_action TEXT, action_type TEXT, direct_interaction INTEGER,
          molecular_mechanism INTEGER, mechanism_comment TEXT,
          selectivity_comment TEXT, binding_site_comment TEXT
        );
        CREATE TABLE mechanism_refs (
          mecref_id INTEGER PRIMARY KEY, mec_id INTEGER, ref_type TEXT,
          ref_id TEXT, ref_url TEXT
        );
        CREATE TABLE target_dictionary (
          tid INTEGER PRIMARY KEY, organism TEXT, target_type TEXT
        );
        CREATE TABLE target_components (
          tid INTEGER, component_id INTEGER
        );
        CREATE TABLE component_sequences (
          component_id INTEGER PRIMARY KEY, accession TEXT
        );
        INSERT INTO target_dictionary VALUES
          (501, 'Homo sapiens', 'SINGLE PROTEIN'),
          (502, 'Homo sapiens', 'SINGLE PROTEIN');
        INSERT INTO target_components VALUES (501, 601), (502, 602);
        INSERT INTO component_sequences VALUES (601, 'P00001'), (602, 'P00002');
        INSERT INTO assays VALUES
          (1, 'human receptor inhibition in HEK293', 'Homo sapiens', 9606,
           NULL, 'skin', 'HEK293', NULL, 10, 'BAO_0000219'),
          (2, 'purified target B binding', NULL, NULL,
           NULL, NULL, NULL, NULL, NULL, 'BAO_0000357');
        INSERT INTO cell_dictionary VALUES
          (10, 'HEK293', 'engineered kidney cell', 'kidney',
           'Homo sapiens', 9606, 'CVCL_0045');
        INSERT INTO molecule_dictionary VALUES
          (100, 'CHEMBL100', 'TEST INHIBITOR'),
          (101, 'CHEMBL101', NULL),
          (200, 'CHEMBL200', 'PARENT');
        INSERT INTO molecule_hierarchy VALUES (100, 200), (101, NULL);
        INSERT INTO compound_structures VALUES
          (100, 'C[C@H](O)F', NULL), (101, 'CCO', NULL), (200, 'CC(O)F', NULL);
        INSERT INTO drug_mechanism VALUES
          (7, 100, 501, 'target A antagonist', 'ANTAGONIST', 1, 1,
           'curated mechanism', NULL, NULL);
        INSERT INTO mechanism_refs VALUES
          (70, 7, 'PubMed', '12345', 'https://pubmed.ncbi.nlm.nih.gov/12345/'),
          (71, 7, 'PubChem', '999', 'https://pubchem.ncbi.nlm.nih.gov/compound/999');
        """
    )
    connection.commit()
    connection.close()


def _chembl_rows() -> pd.DataFrame:
    rows = []
    for index in range(12):
        smiles = "C[C@H](O)F" if index == 0 else "C" * (index + 1) + "O"
        molecule_id = "CHEMBL100" if index == 0 else f"CHEMBL{1000 + index}"
        molregno = 100 if index == 0 else 101
        rows.append(
            {
                "target_chembl_id": "CHEMBL_TA",
                "uniprot": "P00001",
                "molecule_chembl_id": molecule_id,
                "smiles": smiles,
                "source_db": "ChEMBL",
                "source_release": "37",
                "source_license": "CC BY-SA 3.0",
                "source_db_sha256": "a" * 64,
                "activity_id": index + 1,
                "assay_id": 1,
                "assay_chembl_id": "CHEMBL_A1",
                "assay_type": "F",
                "assay_test_type": None,
                "assay_category": None,
                "assay_confidence_score": 9,
                "assay_relationship_type": "D",
                "assay_source_id": 1,
                "assay_source_name": "LITERATURE",
                "target_id": 501,
                "target_organism": "Homo sapiens",
                "target_pref_name": "Target A",
                "molecule_molregno": molregno,
                "standard_inchi_key": Chem.MolToInchiKey(Chem.MolFromSmiles(smiles)),
                "document_chembl_id": "CHEMBL_DOC1",
                "document_year": 2024,
                "pubmed_id": "12345",
                "doi": "10.1000/test",
                "patent_id": None,
                "document_source_name": "LITERATURE",
                "standard_relation": ">" if index == 1 else "=",
                "standard_type": "Ki" if index == 1 else "IC50",
                "standard_value": float(index + 1),
                "standard_units": "nM",
                "pchembl_value": 9.0 - index / 10,
                "data_validity_comment": None,
                "potential_duplicate": False,
                "activity_comment": None,
                "action_type": "INHIBITOR" if index in {0, 1} else None,
            }
        )
    shared = dict(rows[0])
    shared.update(
        {
            "target_chembl_id": "CHEMBL_TB",
            "uniprot": "P00002",
            "activity_id": 100,
            "assay_id": 2,
            "assay_chembl_id": "CHEMBL_A2",
            "target_id": 502,
            "target_pref_name": "Target B",
        }
    )
    rows.append(shared)
    return pd.DataFrame(rows)


def _binding_rows() -> pd.DataFrame:
    smiles = "C[C@H](O)F"
    computed_key = Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))
    unspecified_key = computed_key[:14] + "-UHFFFAOYSA-N"
    return pd.DataFrame(
        [
            {
                "evidence_id": "binding-1",
                "duplicate_group_id": "dup-1",
                "duplicate_evidence": False,
                "input_row_number": 77,
                "source_origin": "Article",
                "source_release": "2026-08",
                "source_license": "CC BY 3.0 (BindingDB-curated)",
                "ligand_smiles": smiles,
                "ligand_inchikey": unspecified_key,
                "ligand_id": "BDB-1",
                "uniprot": "P00001",
                "organism": "Homo sapiens",
                "affinity_type": "Ki",
                "relation": "<",
                "censor": True,
                "affinity_value": 20.0,
                "affinity_unit": "nM",
                "source_pmid": "12345",
                "source_doi": "10.1000/test",
                "source_patent": "",
                "source_article_id": "article-1",
            }
        ]
    )


def _intent(
    intent_id: str,
    *,
    biomarker: str,
    uniprot_id: str | None,
    route: str,
    desired_effect: str,
) -> dict[str, object]:
    return {
        "schema_version": "skinscout.target-intent.v1",
        "intent_id": intent_id,
        "source_row_id": f"row-{intent_id}",
        "source_sheet": "Sheet1",
        "source_row_number": 3,
        "category": "barrier" if route == "endpoint" else "skin",
        "biomarker": biomarker,
        "full_name": biomarker,
        "marker_type": "endpoint" if route == "endpoint" else "protein",
        "role_summary": "fixture",
        "source_direction": "fixture direction",
        "route": route,
        "entity_type": "endpoint" if route == "endpoint" else "protein",
        "entity_name": biomarker,
        "gene_symbol": None if route == "endpoint" else biomarker,
        "uniprot_id": uniprot_id,
        "protein_form": None,
        "component_of": None,
        "desired_effect": desired_effect,
        "target_scope": "human",
        "skin_compartment": "fixture skin compartment",
        "readout": "fixture readout",
        "source_evidence": {
            "workbook_path": "fixture.xlsx",
            "workbook_sha256": "a" * 64,
            "sheet": "Sheet1",
            "row_number": 3,
            "gene_map_path": "gene_map.csv",
            "gene_map_sha256": "b" * 64,
            "registry_path": "registry.csv",
            "registry_sha256": "c" * 64,
            "curated_on": "2026-09-15",
        },
        "target_taxid": None if route == "endpoint" else 9606,
        "role": "material_candidate",
        "docking_eligible": route != "endpoint",
    }


def test_build_retains_full_ledger_identity_provenance_and_zero_coverage(
    tmp_path: Path,
) -> None:
    intents = pd.DataFrame(
        [
            _intent(
                "intent-a",
                biomarker="TA",
                uniprot_id="P00001",
                route="direct_target",
                desired_effect="inhibit_function",
            ),
            _intent(
                "intent-b",
                biomarker="TB",
                uniprot_id="P00002",
                route="direct_target",
                desired_effect="inhibit_function",
            ),
            _intent(
                "intent-endpoint",
                biomarker="TEWL",
                uniprot_id=None,
                route="endpoint",
                desired_effect="decrease_endpoint",
            ),
        ]
    )
    intents_path = tmp_path / "intents.json"
    intents_path.write_text(json.dumps(intents.to_dict("records")), encoding="utf-8")
    gene_map = tmp_path / "gene_map.csv"
    pd.DataFrame([{"gene": "TA", "uniprot": "P00001"}]).to_csv(gene_map, index=False)
    chembl = tmp_path / "chembl.parquet"
    binding = tmp_path / "binding.parquet"
    _chembl_rows().to_parquet(chembl, index=False)
    _binding_rows().to_parquet(binding, index=False)
    database = tmp_path / "chembl.db"
    _create_sqlite(database)
    out_dir = tmp_path / "out"

    completed = subprocess.run(
        [
            str(PYTHON),
            str(SCRIPT),
            "--intents",
            str(intents_path),
            "--gene-map",
            str(gene_map),
            "--chembl-evidence",
            str(chembl),
            "--bindingdb-evidence",
            str(binding),
            "--chembl-sqlite",
            str(database),
            "--out-dir",
            str(out_dir),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    ledger = pd.read_parquet(out_dir / "candidate_evidence.parquet")
    summary = pd.read_parquet(out_dir / "candidate_source_summary.parquet")
    coverage = pd.read_csv(out_dir / "evidence_coverage.csv")
    manifest = json.loads((out_dir / "manifest.json").read_text())

    assert len(ledger) == 15
    assert manifest["counts"]["ledger_rows"] == 15
    assert manifest["counts"]["input_intents"] == 3
    # 12 ChEMBL candidates plus one stereo-scoped BindingDB candidate.
    assert len(summary[summary["intent_id"] == "intent-a"]) == 13
    assert set(ledger[ledger["source_db"] == "BindingDB"]["direction_relation"]) == {
        "unknown"
    }
    assert set(ledger[ledger["source_db"] == "BindingDB"]["evidence_axis"]) == {
        "binding"
    }
    assert ledger.loc[ledger["endpoint_relation"] == ">", "endpoint_censored"].all()

    chembl_first = ledger[
        (ledger["intent_id"] == "intent-a") & (ledger["source_activity_id"] == 1)
    ].iloc[0]
    assert chembl_first["molecule_pref_name"] == "TEST INHIBITOR"
    assert chembl_first["assay_description"] == "human receptor inhibition in HEK293"
    assert chembl_first["cell_origin_taxid"] == 9606
    assert chembl_first["model_level"] == "unknown"
    assert chembl_first["mechanism_direct_interaction"]
    assert "PubChem" in chembl_first["mechanism_records_json"]
    assert chembl_first["compound_id"].startswith("structure-sha256:")
    assert chembl_first["structure_identity_status"] == "validated_match"
    assert chembl_first["parent_source_compound_id"] == "CHEMBL200"
    mechanism = ledger[ledger["raw_evidence_id"] == "mechanism:7"].iloc[0]
    assert mechanism["direction_relation"] == "supports"
    candidate = summary[
        (summary["intent_id"] == "intent-a")
        & (summary["compound_id"] == chembl_first["compound_id"])
    ].iloc[0]
    assert json.loads(candidate["activity_action_types_json"]) == ["INHIBITOR"]
    assert json.loads(candidate["mechanism_action_types_json"]) == ["ANTAGONIST"]
    assert candidate["source_identity_mismatch_count"] == 0
    assert candidate["source_connectivity_mismatch_count"] == 0
    assert candidate["source_stereo_mismatch_count"] == 0
    assert set(json.loads(candidate["structure_identity_statuses_json"])) == {
        "computed_no_source_inchikey",
        "validated_match",
    }
    assert json.loads(candidate["identity_scopes_json"]) == ["absolute_exact"]
    assert len(json.loads(candidate["raw_inchikeys_json"])) == 1

    scoped = summary[summary["compound_id"].str.startswith("stereo-scoped-sha256:")]
    assert len(scoped) == 1
    scoped_candidate = scoped.iloc[0]
    assert scoped_candidate["intent_id"] == "intent-a"
    assert scoped_candidate["source_stereo_mismatch_count"] == 1
    assert scoped_candidate["source_connectivity_mismatch_count"] == 0
    assert json.loads(scoped_candidate["identity_scopes_json"]) == [
        "unspecified_stereo"
    ]
    assert "must not be used as exact" in json.loads(
        scoped_candidate["structure_identity_limitations_json"]
    )[0]

    functional_ki = ledger[
        (ledger["source_activity_id"] == 2) & (ledger["source_db"] == "ChEMBL")
    ].iloc[0]
    assert functional_ki["assay_type"] == "F"
    assert functional_ki["endpoint_type"] == "Ki"
    assert functional_ki["evidence_axis"] == "binding"
    assert functional_ki["direction_relation"] == "unknown"

    shared = ledger[ledger["canonical_isomeric_smiles"] == "C[C@H](O)F"]
    assert set(shared["intent_id"]) == {"intent-a", "intent-b"}
    assert set(shared["identity_scope"]) == {"absolute_exact", "unspecified_stereo"}
    assert shared[shared["identity_scope"] == "absolute_exact"]["compound_id"].nunique() == 1
    assert manifest["counts"]["connectivity_mismatch_rows"] == 0
    assert manifest["counts"]["stereo_mismatch_rows"] == 1
    assert manifest["counts"]["identity_scope_rows"]["unspecified_stereo"] == 1
    assert manifest["counts"]["unique_stereo_scoped_candidate_ids"] == 1
    endpoint = coverage[coverage["intent_id"] == "intent-endpoint"].iloc[0]
    assert endpoint["evidence_count"] == 0
    assert endpoint["evidence_status"] == "no_local_source_evidence"
    assert "nonprotein endpoint" in endpoint["coverage_reason"]
    assert manifest["software"]["builder"]["sha256"]
    assert manifest["software"]["target_intent"]["sha256"]
    assert manifest["software"]["rdkit"]


def test_structure_identity_keeps_stereochemistry_and_reports_mismatch() -> None:
    module = _load_module()
    left = module._structure("C[C@H](O)F", None)
    right = module._structure("C[C@@H](O)F", None)
    assert left["compound_id"] != right["compound_id"]
    mismatch = module._structure("C[C@H](O)F", "AAAAAAAAAAAAAA-BBBBBBBBBB-C")
    assert mismatch["structure_identity_status"] == "source_inchikey_mismatch"


def test_relative_cx_stereo_is_preserved_and_scoped() -> None:
    module = _load_module()
    exact = module._structure("C[C@H](O)F", "OGBOTYGRYZDLMG-REOHCLBHSA-N")
    relative = module._structure("C[C@H](O)F |r|", "OGBOTYGRYZDLMG-UHFFFAOYSA-N")

    assert relative["stereo_chemistry_class"] == "relative"
    assert relative["identity_scope"] == "relative_stereo"
    assert relative["compound_id"].startswith("stereo-scoped-sha256:")
    assert relative["canonical_isomeric_smiles"].endswith("|r|")
    assert relative["structure_identity_status"] == "source_stereo_mismatch"
    assert relative["compound_id"] != exact["compound_id"]
    assert "exact absolute" in relative["structure_identity_limitation"]


def test_unspecified_source_stereo_never_becomes_exact_absolute() -> None:
    module = _load_module()
    computed_key = Chem.MolToInchiKey(Chem.MolFromSmiles("C[C@H](O)F"))
    unspecified = module._structure(
        "C[C@H](O)F", computed_key[:14] + "-UHFFFAOYSA-N"
    )

    assert unspecified["stereo_chemistry_class"] == "unspecified"
    assert unspecified["identity_scope"] == "unspecified_stereo"
    assert unspecified["compound_id"].startswith("stereo-scoped-sha256:")
    assert unspecified["structure_identity_status"] == "source_stereo_mismatch"
    assert unspecified["canonical_isomeric_smiles"] == "C[C@H](O)F"
    assert (
        module._structure("CCO", "LFQSCWFLJHTTHZ-UHFFFAOYSA-N")["identity_scope"]
        == "absolute_exact"
    )


def test_enhanced_stereo_groups_stay_scoped_and_are_written() -> None:
    module = _load_module()
    relative = module._structure("C[C@H](O)[C@@H](C)O |o1:1,3|", None)
    racemic = module._structure("C[C@H](O)[C@@H](C)O |&1:1,3|", None)

    assert relative["identity_scope"] == "relative_stereo"
    assert racemic["identity_scope"] == "racemic_stereo"
    assert "o1:1,3" in relative["canonical_isomeric_smiles"]
    assert "&1:1,3" in racemic["canonical_isomeric_smiles"]


def test_relative_stereo_ledger_rows_are_scoped_with_separate_counters(
    tmp_path: Path,
) -> None:
    module = _load_module()
    intent = _intent(
        "intent-a",
        biomarker="TA",
        uniprot_id="P00001",
        route="direct_target",
        desired_effect="inhibit_function",
    )
    intents_path = tmp_path / "intents.json"
    intents_path.write_text(json.dumps([intent]), encoding="utf-8")
    gene_map = tmp_path / "gene_map.csv"
    pd.DataFrame([{"gene": "TA", "uniprot": "P00001"}]).to_csv(gene_map, index=False)
    chembl = tmp_path / "chembl.parquet"
    _chembl_rows().to_parquet(chembl, index=False)
    exact_smiles = "C[C@H](O)F"
    exact_key = Chem.MolToInchiKey(Chem.MolFromSmiles(exact_smiles))
    base = _binding_rows().iloc[0].to_dict()
    binding = tmp_path / "binding.parquet"
    pd.DataFrame(
        [
            {
                **base,
                "evidence_id": "binding-exact",
                "ligand_id": "BDB-EXACT",
                "ligand_smiles": exact_smiles,
                "ligand_inchikey": exact_key,
            },
            {
                **base,
                "evidence_id": "binding-relative",
                "ligand_id": "BDB-REL",
                "ligand_smiles": f"{exact_smiles} |r|",
                "ligand_inchikey": exact_key[:14] + "-UHFFFAOYSA-N",
            },
        ]
    ).to_parquet(binding, index=False)
    database = tmp_path / "chembl.db"
    _create_sqlite(database)
    out_dir = tmp_path / "out"

    manifest = module.build(
        module.parse_args(
            [
                "--intents",
                str(intents_path),
                "--gene-map",
                str(gene_map),
                "--chembl-evidence",
                str(chembl),
                "--bindingdb-evidence",
                str(binding),
                "--chembl-sqlite",
                str(database),
                "--out-dir",
                str(out_dir),
            ]
        )
    )

    ledger = pd.read_parquet(out_dir / "candidate_evidence.parquet")
    relative_row = ledger[ledger["raw_smiles"] == f"{exact_smiles} |r|"].iloc[0]
    assert relative_row["compound_id"].startswith("stereo-scoped-sha256:")
    assert relative_row["stereo_chemistry_class"] == "relative"
    assert relative_row["identity_scope"] == "relative_stereo"
    assert relative_row["canonical_isomeric_smiles"].endswith("|r|")
    assert relative_row["structure_identity_status"] == "source_stereo_mismatch"
    assert "must not be used as exact" in relative_row["structure_identity_limitation"]
    exact_row = ledger[
        (ledger["raw_smiles"] == exact_smiles)
        & (ledger["compound_id"].str.startswith("structure-sha256:"))
    ].iloc[0]
    assert exact_row["identity_scope"] == "absolute_exact"
    assert manifest["counts"]["connectivity_mismatch_rows"] == 0
    assert manifest["counts"]["stereo_mismatch_rows"] >= 1
    assert manifest["counts"]["identity_scope_rows"].get("relative_stereo", 0) >= 1


def test_ki_and_kd_never_become_functional_from_assay_type_or_action() -> None:
    module = _load_module()
    for endpoint in ("Ki", "Kd"):
        assert (
            module._chembl_evidence_axis(
                {"standard_type": endpoint, "assay_type": "F"}, "INHIBITOR"
            )
            == "binding"
        )
    assert (
        module._model_level({"assay_type": "B", "description": "binding assay"})
        == "unknown"
    )
    assert (
        module._model_level(
            {"assay_type": "B", "description": "purified recombinant protein binding"}
        )
        == "purified_protein"
    )
