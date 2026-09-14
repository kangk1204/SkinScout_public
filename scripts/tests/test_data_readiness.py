"""Regression tests for Stage 0 data readiness preflight."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import networkx as nx
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
READINESS = ROOT / "scripts/data_readiness.py"


def load_readiness_module():
    spec = importlib.util.spec_from_file_location("data_readiness_under_test", READINESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_required_target_id_artifacts_include_stage0_infrastructure(
    tmp_path: Path,
) -> None:
    readiness = load_readiness_module()
    fake_config = {
        "paths": {
            "alphafold_clean": str(tmp_path / "clean"),
            "docking_boxes": str(tmp_path / "boxes"),
            "pdbqt": str(tmp_path / "pdbqt"),
            "no_pocket_list": str(tmp_path / "no_pocket_targets.list"),
        },
        "paths_v3": {
            "cosing": str(tmp_path / "cosing"),
            "drug_avoidance": str(tmp_path / "drug_avoidance"),
            "skin_expression": str(tmp_path / "skin_expression"),
            "skin_kg": str(tmp_path / "skin_kg"),
        },
    }

    artifacts = readiness.required_data_artifacts(
        "target-id",
        "comprehensive",
        config=fake_config,
        root=tmp_path,
    )
    labels = {label for label, _path in artifacts}

    assert "CosIng reference" in labels
    assert "drug reference" in labels
    assert "cleaned AlphaFold receptor marker" in labels
    assert "PDBQT receptor marker" in labels
    assert "docking box directory" in labels
    assert "skin-expression score table" in labels
    assert "skin efficacy KG" in labels


def test_artifact_missing_treats_empty_files_and_directories_as_missing(
    tmp_path: Path,
) -> None:
    readiness = load_readiness_module()
    missing = tmp_path / "missing.txt"
    empty_file = tmp_path / "empty.txt"
    nonempty_file = tmp_path / "present.txt"
    empty_dir = tmp_path / "empty_dir"
    nonempty_dir = tmp_path / "nonempty_dir"

    empty_file.write_text("")
    nonempty_file.write_text("x")
    empty_dir.mkdir()
    nonempty_dir.mkdir()
    (nonempty_dir / "child").write_text("x")

    assert readiness.artifact_missing(missing) is True
    assert readiness.artifact_missing(empty_file) is True
    assert readiness.artifact_missing(empty_dir) is True
    assert readiness.artifact_missing(nonempty_file) is False
    assert readiness.artifact_missing(nonempty_dir) is False


def test_data_readiness_payload_reports_missing_artifacts(tmp_path: Path) -> None:
    readiness = load_readiness_module()
    fake_config = {
        "paths": {
            "alphafold_clean": str(tmp_path / "clean"),
            "docking_boxes": str(tmp_path / "boxes"),
            "pdbqt": str(tmp_path / "pdbqt"),
            "no_pocket_list": str(tmp_path / "no_pocket_targets.list"),
        },
        "paths_v3": {
            "cosing": str(tmp_path / "cosing"),
            "drug_avoidance": str(tmp_path / "drug_avoidance"),
            "skin_expression": str(tmp_path / "skin_expression"),
            "skin_kg": str(tmp_path / "skin_kg"),
        },
    }

    payload = readiness.data_readiness_payload(
        "target-id",
        "comprehensive",
        True,
        config=fake_config,
        root=tmp_path,
    )

    assert payload["status"] == "failed"
    missing = readiness.missing_from_payload(payload)
    assert any(
        label.startswith("CosIng reference [missing]")
        and path == tmp_path / "cosing/cosing.parquet"
        for label, path in missing
    )
    assert any(label.startswith("docking box directory [missing]") for label, _path in missing)


def test_required_stage0_source_artifacts_default_to_no_operator_sources(
    tmp_path: Path,
) -> None:
    readiness = load_readiness_module()
    fake_config = {
        "cosmetic_db_mode": "full",
        "stage0": {"require_drugbank_source": False},
        "cosing": {"auto_mirror_public_api": True},
        "skin_expression": {
            "require_proteome_source": False,
            "require_gtex_source": False,
            "allow_empty_sources": False,
        },
        "paths": {"drugbank": str(tmp_path / "drugbank")},
        "paths_v3": {
            "cosing": str(tmp_path / "cosing"),
            "skin_proteome": str(tmp_path / "skin_proteome"),
            "gtex": str(tmp_path / "gtex"),
        },
    }

    artifacts = readiness.required_data_artifacts(
        "stage0",
        "comprehensive",
        config=fake_config,
        root=tmp_path,
    )

    assert artifacts == []


def test_required_stage0_sources_include_cosing_when_auto_mirror_disabled(
    tmp_path: Path,
) -> None:
    readiness = load_readiness_module()
    fake_config = {
        "cosmetic_db_mode": "full",
        "stage0": {"require_drugbank_source": False},
        "cosing": {"auto_mirror_public_api": False},
        "skin_expression": {
            "require_proteome_source": False,
            "require_gtex_source": False,
            "allow_empty_sources": False,
        },
        "paths": {"drugbank": str(tmp_path / "drugbank")},
        "paths_v3": {
            "cosing": str(tmp_path / "cosing"),
            "skin_proteome": str(tmp_path / "skin_proteome"),
            "gtex": str(tmp_path / "gtex"),
        },
    }

    artifacts = readiness.required_data_artifacts(
        "stage0",
        "comprehensive",
        config=fake_config,
        root=tmp_path,
    )

    assert artifacts == [
        (
            "Stage 0 source: CosIng CSV",
            tmp_path / "cosing/cosing.csv",
        ),
    ]


def test_required_stage0_sources_include_skin_enrichment_when_required(
    tmp_path: Path,
) -> None:
    readiness = load_readiness_module()
    fake_config = {
        "cosmetic_db_mode": "placeholder",
        "stage0": {"require_drugbank_source": False},
        "skin_expression": {
            "require_proteome_source": True,
            "require_gtex_source": True,
            "allow_empty_sources": False,
        },
        "paths": {"drugbank": str(tmp_path / "drugbank")},
        "paths_v3": {
            "cosing": str(tmp_path / "cosing"),
            "skin_proteome": str(tmp_path / "skin_proteome"),
            "gtex": str(tmp_path / "gtex"),
        },
    }

    artifacts = readiness.required_data_artifacts(
        "stage0",
        "comprehensive",
        config=fake_config,
        root=tmp_path,
    )

    assert artifacts == [
        (
            "Stage 0 source: skin proteome LFQ TSV",
            tmp_path / "skin_proteome/raw_lfq.tsv",
        ),
        (
            "Stage 0 source: GTEx gene TPM GCT",
            tmp_path / "gtex/gtex_v10_gene_tpm.gct",
        ),
    ]


def test_required_stage0_sources_include_drugbank_when_required(
    tmp_path: Path,
) -> None:
    readiness = load_readiness_module()
    fake_config = {
        "cosmetic_db_mode": "placeholder",
        "stage0": {
            "require_drugbank_source": True,
            "allow_drugbank_placeholder": False,
        },
        "skin_expression": {"allow_empty_sources": True},
        "paths": {"drugbank": str(tmp_path / "drugbank")},
        "paths_v3": {
            "cosing": str(tmp_path / "cosing"),
            "skin_proteome": str(tmp_path / "skin_proteome"),
            "gtex": str(tmp_path / "gtex"),
        },
    }

    artifacts = readiness.required_data_artifacts(
        "stage0",
        "comprehensive",
        config=fake_config,
        root=tmp_path,
    )

    assert artifacts == [
        (
            "Stage 0 source: DrugBank full database XML",
            tmp_path / "drugbank/drugbank_full_database.xml",
        )
    ]


def test_optional_stage0_sources_list_enrichments_when_not_required(
    tmp_path: Path,
) -> None:
    readiness = load_readiness_module()
    fake_config = {
        "cosmetic_db_mode": "full",
        "stage0": {"require_drugbank_source": False},
        "cosing": {"auto_mirror_public_api": True},
        "skin_expression": {
            "require_proteome_source": False,
            "require_gtex_source": False,
        },
        "paths": {"drugbank": str(tmp_path / "drugbank")},
        "paths_v3": {
            "cosing": str(tmp_path / "cosing"),
            "skin_proteome": str(tmp_path / "skin_proteome"),
            "gtex": str(tmp_path / "gtex"),
        },
    }

    assert readiness.optional_stage0_source_artifacts(
        config=fake_config,
        root=tmp_path,
    ) == [
        (
            "Stage 0 source: CosIng CSV",
            tmp_path / "cosing/cosing.csv",
        ),
        (
            "Stage 0 source: DrugBank full database XML",
            tmp_path / "drugbank/drugbank_full_database.xml",
        ),
        (
            "Stage 0 source: skin proteome LFQ TSV",
            tmp_path / "skin_proteome/raw_lfq.tsv",
        ),
        (
            "Stage 0 source: GTEx gene TPM GCT",
            tmp_path / "gtex/gtex_v10_gene_tpm.gct",
        ),
    ]


def test_required_skin_enrichments_are_not_optional_sources(
    tmp_path: Path,
) -> None:
    readiness = load_readiness_module()
    fake_config = {
        "stage0": {"require_drugbank_source": False},
        "skin_expression": {
            "require_proteome_source": True,
            "require_gtex_source": True,
        },
        "paths": {"drugbank": str(tmp_path / "drugbank")},
        "paths_v3": {
            "skin_proteome": str(tmp_path / "skin_proteome"),
            "gtex": str(tmp_path / "gtex"),
        },
    }

    assert readiness.optional_stage0_source_artifacts(
        config=fake_config,
        root=tmp_path,
    ) == [
        (
            "Stage 0 source: DrugBank full database XML",
            tmp_path / "drugbank/drugbank_full_database.xml",
        )
    ]


def test_stage0_source_missing_uses_source_failure_message(tmp_path: Path) -> None:
    readiness = load_readiness_module()
    missing = [
        (
            "Stage 0 source: CosIng CSV [missing]",
            tmp_path / "cosing/cosing.csv",
        )
    ]

    message = readiness.failure_message(missing)

    assert "Stage 0 source readiness preflight failed" in message
    assert "--allow-stage0-build" not in message


def test_stage0_source_readiness_rejects_malformed_gtex_gct(
    tmp_path: Path,
) -> None:
    readiness = load_readiness_module()
    gct = tmp_path / "gtex_v10_gene_tpm.gct"
    gct.write_text(
        "#1.2\n"
        "1\t3\n"
        "Name\tDescription\tMuscle sample\n"
        "ENSG000001\tGENE1\t1.0\n"
    )

    check = readiness.check_required_artifact(
        "Stage 0 source: GTEx gene TPM GCT",
        gct,
    )

    assert check.status == "invalid"
    assert "contains no skin sample columns" in check.detail


def test_stage0_source_readiness_accepts_valid_cosing_csv(
    tmp_path: Path,
) -> None:
    readiness = load_readiness_module()
    cosing = tmp_path / "cosing.csv"
    cosing.write_text("INCI name;CAS;Function\nWater;7732-18-5;Solvent\n")

    check = readiness.check_required_artifact("Stage 0 source: CosIng CSV", cosing)

    assert check.status == "present"
    assert "nonblank_inci_names=1" in check.detail


def test_data_readiness_rejects_placeholder_cosing_reference(tmp_path: Path) -> None:
    readiness = load_readiness_module()
    cosing = tmp_path / "cosing.parquet"
    pd.DataFrame(
        [
            {
                "inci_name": "Niacinamide",
                "smiles": "NC(=O)c1cccnc1",
                "inchikey": "DFPAKSUCGFBDDF-UHFFFAOYSA-N",
                "ecfp4": [0] * 32,
                "scaffold_smiles": "c1ccncc1",
            }
        ]
    ).to_parquet(cosing)

    check = readiness.check_required_artifact("CosIng reference", cosing)

    assert check.status == "placeholder"
    assert "inci_name=Niacinamide" in check.detail


def test_data_readiness_accepts_full_cosing_reference_with_common_ingredients(
    tmp_path: Path,
) -> None:
    readiness = load_readiness_module()
    cosing = tmp_path / "cosing.parquet"
    rows = [
        {
            "inci_name": "Niacinamide" if idx == 0 else f"Ingredient {idx}",
            "smiles": "CCO",
            "inchikey": f"KEY{idx:04d}",
            "ecfp4": [idx % 2] * 32,
            "scaffold_smiles": "CCO",
        }
        for idx in range(100)
    ]
    pd.DataFrame(rows).to_parquet(cosing)

    check = readiness.check_required_artifact("CosIng reference", cosing)

    assert check.status == "present"


def test_data_readiness_rejects_placeholder_drug_reference(tmp_path: Path) -> None:
    readiness = load_readiness_module()
    drugs = tmp_path / "drugs.parquet"
    pd.DataFrame(
        [
            {
                "drug_id": "PLACEHOLDER1",
                "name": "Aspirin",
                "smiles": "CC(=O)Oc1ccccc1C(=O)O",
                "inchikey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
                "ecfp4": [0] * 32,
            }
        ]
    ).to_parquet(drugs)

    check = readiness.check_required_artifact("drug reference", drugs)

    assert check.status == "placeholder"
    assert "drug_id=PLACEHOLDER1" in check.detail


def test_data_readiness_accepts_full_drug_reference_with_common_drug_names(
    tmp_path: Path,
) -> None:
    readiness = load_readiness_module()
    drugs = tmp_path / "drugs.parquet"
    rows = [
        {
            "drug_id": f"DRUG{idx:04d}",
            "name": "Aspirin" if idx == 0 else "Ibuprofen" if idx == 1 else f"Drug {idx}",
            "smiles": "CCO",
            "inchikey": f"DRUGKEY{idx:04d}",
            "ecfp4": [idx % 2] * 32,
        }
        for idx in range(100)
    ]
    pd.DataFrame(rows).to_parquet(drugs)

    check = readiness.check_required_artifact("drug reference", drugs)

    assert check.status == "present"


def test_data_readiness_rejects_too_small_scaffold_reference(tmp_path: Path) -> None:
    readiness = load_readiness_module()
    scaffolds = tmp_path / "scaffolds.parquet"
    pd.DataFrame([{"scaffold_smiles": "c1ccccc1", "n_drugs": 2}]).to_parquet(scaffolds)

    check = readiness.check_required_artifact("drug scaffold reference", scaffolds)

    assert check.status == "too_small"
    assert "rows=1 min_rows=20" in check.detail


def test_data_readiness_rejects_seed_only_skin_kg(tmp_path: Path) -> None:
    readiness = load_readiness_module()
    graph_path = tmp_path / "skin_efficacy.graphml"
    graph = nx.MultiDiGraph()
    graph.add_node("category:hydration", type="EfficacyCategory", name="hydration")
    graph.add_node("gene:P20930", type="Gene", uniprot="P20930")
    graph.add_edge("gene:P20930", "category:hydration", relation="ASSOCIATED_WITH")
    nx.write_graphml(graph, graph_path)

    check = readiness.check_required_artifact("skin efficacy KG", graph_path)

    assert check.status == "placeholder"
    assert "pubtator_backed_nodes=0" in check.detail


def test_data_readiness_stage0_degraded_config_has_no_source_checks(
    tmp_path: Path,
) -> None:
    readiness = load_readiness_module()
    fake_config = {
        "cosmetic_db_mode": "placeholder",
        "stage0": {"allow_drugbank_placeholder": True},
        "skin_expression": {"allow_empty_sources": True},
        "paths": {"drugbank": str(tmp_path / "drugbank")},
        "paths_v3": {
            "cosing": str(tmp_path / "cosing"),
            "skin_proteome": str(tmp_path / "skin_proteome"),
            "gtex": str(tmp_path / "gtex"),
        },
    }

    payload = readiness.data_readiness_payload(
        "stage0",
        "comprehensive",
        True,
        config=fake_config,
        root=tmp_path,
    )

    assert payload["status"] == "ok"
    assert payload["checks"] == []


def test_data_readiness_json_out_matches_stdout(tmp_path: Path) -> None:
    out = tmp_path / "data_readiness.json"

    res = subprocess.run(
        [
            sys.executable,
            str(READINESS),
            "--json",
            "--allow-stage0-build",
            "--json-out",
            str(out),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode == 0
    assert json.loads(out.read_text()) == json.loads(res.stdout)


def test_allow_stage0_build_message_is_not_a_failure() -> None:
    readiness = load_readiness_module()

    message = readiness.allow_stage0_build_message(
        [("missing marker", Path("/tmp/missing.marker"))]
    )

    assert "--allow-stage0-build was set" in message
    assert "Data readiness preflight failed" not in message


def test_allow_stage0_build_applies_only_to_missing_generated_outputs() -> None:
    readiness = load_readiness_module()

    assert readiness.failed_entries_are_stage0_buildable(
        [("PDBQT receptor marker [missing]", Path("/tmp/.pdbqt_complete"))]
    )
    assert not readiness.failed_entries_are_stage0_buildable(
        [("CosIng reference [placeholder] rows=3", Path("/tmp/cosing.parquet"))]
    )
    assert not readiness.failed_entries_are_stage0_buildable(
        [("drug scaffold reference [too_small] rows=1", Path("/tmp/scaffolds.parquet"))]
    )
    assert not readiness.failed_entries_are_stage0_buildable(
        [("Stage 0 source: CosIng CSV [missing]", Path("/tmp/cosing.csv"))]
    )


def test_stage0_source_failures_for_build_validate_operator_sources(
    tmp_path: Path,
) -> None:
    readiness = load_readiness_module()
    fake_config = {
        "cosmetic_db_mode": "full",
        "stage0": {"allow_drugbank_placeholder": True},
        "cosing": {"auto_mirror_public_api": False},
        "skin_expression": {"allow_empty_sources": True},
        "paths": {"drugbank": str(tmp_path / "drugbank")},
        "paths_v3": {
            "cosing": str(tmp_path / "cosing"),
            "skin_proteome": str(tmp_path / "skin_proteome"),
            "gtex": str(tmp_path / "gtex"),
        },
    }

    missing = readiness.stage0_source_failures_for_build(
        config=fake_config,
        root=tmp_path,
    )

    assert missing == [
        (
            "Stage 0 source: CosIng CSV [missing]",
            tmp_path / "cosing/cosing.csv",
        )
    ]


def test_empty_stage0_marker_file_is_present(tmp_path: Path) -> None:
    readiness = load_readiness_module()
    marker = tmp_path / ".clean_complete"
    marker.touch()

    check = readiness.check_required_artifact(
        "cleaned AlphaFold receptor marker",
        marker,
    )

    assert check.status == "present"
    assert check.size_bytes == 0
