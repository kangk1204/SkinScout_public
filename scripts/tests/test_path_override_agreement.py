"""C06: path overrides must mean one set of paths to DAG, readiness and verifier.

The launcher merges YAML + dotted CLI overrides exactly once; these tests
build two fixture trees (default-ready and custom-ready) and check that
readiness judges only the configured set, that the Snakemake command and the
preflight agree on the merged value, and that stage0_verify resolves the same
override instead of falling back to repo/data.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest
import yaml
from Bio.SeqUtils.CheckSum import crc64

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import data_readiness
import run_skinscout as runner
import stage0_verify
from data_readiness import (
    apply_config_overrides,
    load_workflow_config,
    resolve_stage0_paths,
)

READINESS = SCRIPTS / "data_readiness.py"
PDBQT_MANIFEST = "receptor_prep_manifest.json"
DIGEST = "0" * 64


def _write_pdbqt_manifest(
    pdbqt_dir: Path,
    *,
    expected: int = 1,
    success: int = 1,
    policy: float = 0.8,
    failed_reasons: dict[str, str] | None = None,
) -> Path:
    pdbqt_dir.mkdir(parents=True, exist_ok=True)
    targets = []
    for index in range(success):
        target = f"P{index:05d}"
        (pdbqt_dir / f"{target}.pdbqt").write_text("REMARK ok\n")
        targets.append(
            {
                "uniprot": target,
                "status": "ok",
                "method": "meeko",
                "reason": None,
                "clean_pdb_sha256": DIGEST,
                "pocket_json_sha256": DIGEST,
            }
        )
    for target, reason in sorted((failed_reasons or {}).items()):
        targets.append(
            {
                "uniprot": target,
                "status": "failed",
                "method": None,
                "reason": reason,
                "clean_pdb_sha256": DIGEST,
                "pocket_json_sha256": DIGEST,
            }
        )
    failed_count = len(targets) - success
    assert expected == success + failed_count
    manifest = pdbqt_dir / PDBQT_MANIFEST
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "skinscout.stage0-receptor-pdbqt.v1",
                "scope": "full",
                "status": "ok",
                "policy": {"min_success_fraction": policy, "min_success_count": 1},
                "counts": {
                    "expected": expected,
                    "success": success,
                    "failed": failed_count,
                },
                "targets": targets,
            }
        )
        + "\n"
    )
    return manifest


def write_ready_tree(data: Path) -> None:
    cosing = data / "cosing"
    cosing.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "inci_name": f"Ingredient {idx}",
                "smiles": "CCO",
                "inchikey": f"KEY{idx}",
                "ecfp4": "fp",
                "scaffold_smiles": "CC",
            }
            for idx in range(100)
        ]
    ).to_parquet(cosing / "cosing.parquet")

    drugs = data / "drug_avoidance"
    drugs.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "drug_id": f"DRUG{idx}",
                "name": f"Drug {idx}",
                "smiles": "CCO",
                "inchikey": f"DRUGKEY{idx}",
                "ecfp4": "fp",
            }
            for idx in range(100)
        ]
    ).to_parquet(drugs / "drugs.parquet")
    pd.DataFrame(
        [{"scaffold_smiles": f"C{idx}", "n_drugs": 1} for idx in range(20)]
    ).to_parquet(drugs / "scaffolds.parquet")

    human_clean = data / "human_clean"
    human_clean.mkdir(parents=True)
    (human_clean / ".clean_complete").write_text("")
    _write_pdbqt_manifest(data / "human_pdbqt")

    boxes = data / "boxes"
    boxes.mkdir(parents=True)
    (boxes / "P00000_box.txt").write_text("0 0 0 10 10 10\n")
    (data / "no_pocket_targets.list").write_text("P99999\n")

    skin_expression = data / "skin_expression"
    skin_expression.mkdir(parents=True)
    pd.DataFrame(
        [{"uniprot": f"P{idx:05d}", "skin_score": 0.5} for idx in range(1000)]
    ).to_csv(skin_expression / "skin_score.tsv", sep="\t", index=False)
    (skin_expression / "skin_score.tsv.axes.json").write_text(
        json.dumps(
            {
                "schema_version": "skinscout.skin-score-axes.v2",
                "score_semantics": (
                    "relative_within_build_expression_context"
                ),
                "normalization": (
                    "per-axis min-max across proteins in this build"
                ),
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
        )
        + "\n"
    )

    graph = nx.Graph()
    for idx in range(50):
        gene = f"gene:P{idx:05d}"
        graph.add_node(gene, pmid_count=1)
        graph.add_node(f"efficacy:{idx}")
        graph.add_edge(gene, f"efficacy:{idx}")
    skin_kg = data / "skin_efficacy_kg"
    skin_kg.mkdir(parents=True)
    nx.write_graphml(graph, skin_kg / "skin_efficacy.graphml")

    mmseqs = data / "mmseqs"
    mmseqs.mkdir(parents=True)
    (human_clean / "P00000_clean.pdb").write_text("ATOM\n")
    fasta = mmseqs / "human_canonical.fasta"
    fasta_bytes = b">P00000\nACDEFG\n"
    fasta.write_bytes(fasta_bytes)
    (mmseqs / "human_canonical.fasta.manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "skinscout.afdb-v4-canonical-sequences.v1",
                "provenance": {
                    "source": (
                        "AlphaFold Protein Structure Database human proteome v4 "
                        "raw mmCIF files"
                    ),
                    "derivation": (
                        "Canonical sequences assembled offline from AFDB "
                        "fragment metadata and validated against the full "
                        "UniProt CRC64 embedded by AFDB; this is not an "
                        "independently downloaded UniProt FASTA snapshot."
                    ),
                    "alphafold_directory": str(
                        (data / "alphafold_human_v4").resolve()
                    ),
                    "source_pattern": "*.cif.gz",
                },
                "validation": {
                    "taxonomy_id": "9606",
                    "checksum_algorithm": "Bio.SeqUtils.CheckSum.crc64",
                    "fragment_policy": (
                        "all fragments; exact spans; equal overlaps; "
                        "gapless from residue 1"
                    ),
                    "unknown_residues_allowed": False,
                },
                "counts": {
                    "source_files": 1,
                    "canonical_sequences": 1,
                    "required_receptors": 1,
                    "required_receptors_covered": 1,
                },
                "required_receptor_coverage": {
                    "directory": str(human_clean.resolve()),
                    "missing_accessions": [],
                },
                "artifact": {
                    "path": str(fasta.resolve()),
                    "sha256": hashlib.sha256(fasta_bytes).hexdigest(),
                    "bytes": len(fasta_bytes),
                    "sequence_count": 1,
                },
                "sequences": [
                    {
                        "accession": "P00000",
                        "length": 6,
                        "sequence_sha256": hashlib.sha256(
                            b"ACDEFG"
                        ).hexdigest(),
                        "uniprot_crc64": crc64("ACDEFG").removeprefix("CRC-"),
                        "source_fragments": [
                            {
                                "path": "AF-P00000-F1-model_v4.cif.gz",
                                "compressed_sha256": "0" * 64,
                                "seq_db_align_begin": 1,
                                "seq_db_align_end": 6,
                            }
                        ],
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def config_for(data: Path) -> dict:
    config = load_workflow_config()
    apply_config_overrides(
        config,
        [
            f"paths.alphafold_clean={data / 'human_clean'}",
            f"paths.pdbqt={data / 'human_pdbqt'}",
            f"paths.docking_boxes={data / 'boxes'}",
            f"paths.no_pocket_list={data / 'no_pocket_targets.list'}",
            f"paths.mmseqs={data / 'mmseqs'}",
            f"paths.uniprot_human_fasta={data / 'mmseqs' / 'human_canonical.fasta'}",
            f"paths_v3.cosing={data / 'cosing'}",
            f"paths_v3.drug_avoidance={data / 'drug_avoidance'}",
            f"paths_v3.skin_expression={data / 'skin_expression'}",
            f"paths_v3.skin_kg={data / 'skin_efficacy_kg'}",
        ],
    )
    return config


def test_effective_config_resolves_overrides_once(tmp_path: Path) -> None:
    custom = tmp_path / "custom"
    args = runner.parse_args(
        [
            "--smiles",
            "CCO",
            "--run-id",
            "override_case",
            "--extra-config",
            f"paths.pdbqt={custom}",
            "--extra-config",
            f"paths.alphafold_clean={custom / 'clean'}",
        ]
    )

    config = runner._effective_workflow_config(args)
    paths = resolve_stage0_paths(config)

    assert paths["pdbqt"] == custom
    assert paths["alphafold_clean"] == custom / "clean"
    assert paths["docking_boxes"] == ROOT / "data/docking_boxes_derived"

    command = runner.build_snakemake_command(args, validate_conda_frontend=False)
    command_text = " ".join(command)
    assert f"paths.pdbqt={custom}" in command_text or f"'{custom}'" in command_text


def test_readiness_judges_only_the_configured_tree(tmp_path: Path) -> None:
    default_data = tmp_path / "default" / "data"
    custom_data = tmp_path / "custom" / "data"
    write_ready_tree(default_data)
    write_ready_tree(custom_data)
    config = config_for(custom_data)

    # custom tree is complete: default tree state must not matter
    import shutil

    shutil.rmtree(default_data / "boxes")
    runner._run_data_readiness(
        "target-id",
        "fast",
        False,
        allow_stage0_build=False,
        config=config,
    )

    # default tree is complete again, custom is broken: only the custom path fails
    (default_data / "boxes").mkdir()
    (default_data / "boxes" / "P00000_box.txt").write_text("0 0 0 10 10 10\n")
    shutil.rmtree(custom_data / "boxes")
    with pytest.raises(SystemExit) as exc:
        runner._run_data_readiness(
            "target-id",
            "fast",
            False,
            allow_stage0_build=False,
            config=config,
        )

    message = str(exc.value)
    assert str(custom_data / "boxes") in message
    assert str(default_data / "boxes") not in message


def test_data_readiness_cli_merges_extra_config(tmp_path: Path) -> None:
    cosing_dir = tmp_path / "custom_cosing"
    base = [
        sys.executable,
        str(READINESS),
        "--preset",
        "stage0",
        "--json",
        "--extra-config",
        "cosing.auto_mirror_public_api=false",
        "--extra-config",
        f"paths_v3.cosing={cosing_dir}",
    ]

    missing = subprocess.run(base, cwd=ROOT, text=True, capture_output=True, check=False)
    payload = json.loads(missing.stdout)
    assert payload["status"] == "failed"
    assert any(
        check["path"] == str(cosing_dir / "cosing.csv")
        for check in payload["checks"]
    )

    cosing_dir.mkdir()
    (cosing_dir / "cosing.csv").write_text(
        "INCI name;CAS;Function\nWater;7732-18-5;Solvent\n"
    )
    present = subprocess.run(base, cwd=ROOT, text=True, capture_output=True, check=False)
    assert present.returncode == 0
    assert json.loads(present.stdout)["status"] == "ok"


def test_stage0_verify_resolves_the_same_override(tmp_path: Path) -> None:
    custom_pdbqt = tmp_path / "custom_pdbqt"
    custom_pdbqt.mkdir()
    (custom_pdbqt / "P00000.pdbqt").write_text("REMARK ok\n")
    default_pdbqt = tmp_path / "data" / "human_pdbqt"
    default_pdbqt.mkdir(parents=True)
    (default_pdbqt / "P00000.pdbqt").write_text("BROKEN\n")

    checks = stage0_verify.collect_checks(
        tmp_path,
        check_stage0_flag=False,
        paths={"pdbqt": custom_pdbqt},
    )
    by_name = {check.name: check for check in checks}

    assert by_name["pdbqt_roundtrip"].ok, by_name["pdbqt_roundtrip"].detail

    default_checks = stage0_verify.collect_checks(
        tmp_path,
        check_stage0_flag=False,
    )
    default_by_name = {check.name: check for check in default_checks}
    assert not default_by_name["pdbqt_roundtrip"].ok


def test_snakemake_rule_passes_resolved_paths_to_the_verifier() -> None:
    rule_text = (ROOT / "workflow" / "rules" / "stage0_infra.smk").read_text()

    assert "STAGE0_VERIFY_OVERRIDES" in rule_text
    assert "STAGE0_VERIFY_PATH_KEYS" in rule_text
    assert "STAGE0_VERIFY_V3_PATH_KEYS" in rule_text
    assert "MANIFEST_FILENAME" in rule_text
    assert "{params.verify_paths}" in rule_text
    assert "--pdbqt-min-success-fraction" in rule_text
    assert "--pdbqt-min-success-count" in rule_text
    assert "output.manifest" in rule_text

    config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
    assert config["paths"]["docking_boxes"] == "data/docking_boxes_derived"


def test_default_verifier_paths_still_point_at_repo_data(tmp_path: Path) -> None:
    paths = data_readiness.default_stage0_paths(tmp_path)

    assert paths["pdbqt"] == tmp_path / "data" / "human_pdbqt"
    assert paths["alphafold_clean"] == tmp_path / "data" / "human_clean"
    assert paths["docking_boxes"] == tmp_path / "data" / "docking_boxes_derived"
