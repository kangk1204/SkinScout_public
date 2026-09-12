from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "eval/build_rcsb_holo_panel.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_json(payload: object) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_gz_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with gzip.GzipFile(filename="", mode="wb", fileobj=path.open("wb"), mtime=0) as handle:
        for row in rows:
            handle.write((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode())


def _aspirin() -> tuple[str, str, str]:
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    key = Chem.MolToInchiKey(mol)
    inchi = Chem.MolToInchi(mol)
    return smiles, key, inchi


def _raw_entry(
    entry_id: str,
    *,
    uniprot: str = "P11111",
    comp_id: str = "ASP",
    smiles: str | None = None,
    smiles_stereo: str | None = None,
    inchi: str | None = None,
    inchikey: str | None = None,
    subject: bool = True,
    neighbor_targets: list[str] | None = None,
    residue_count: int = 3,
    struct_conn: bool = False,
    release: str = "2025-02-01",
) -> dict[str, object]:
    aspirin_smiles, aspirin_key, aspirin_inchi = _aspirin()
    smiles = aspirin_smiles if smiles is None else smiles
    smiles_stereo = smiles if smiles_stereo is None else smiles_stereo
    inchi = aspirin_inchi if inchi is None else inchi
    inchikey = aspirin_key if inchikey is None else inchikey
    neighbor_targets = neighbor_targets or ["T"]
    neighbors = []
    for idx in range(1, residue_count + 1):
        for target_asym in neighbor_targets:
            neighbors.append(
                {
                    "distance": 3.0 + idx / 10,
                    "target_asym_id": target_asym,
                    "target_auth_seq_id": str(idx),
                    "target_seq_id": idx,
                    "target_comp_id": "ALA",
                    "target_entity_id": "1",
                }
            )
    annotations = []
    if subject:
        annotations.append(
            {
                "scope": "nonpolymer_entity",
                "rcsb_id": f"{entry_id}_2",
                "annotation": {
                    "type": "SUBJECT_OF_INVESTIGATION",
                    "provenance_source": "PDB",
                },
            }
        )
    mappings = [
        {
            "entity_id": "1",
            "instance_id": f"{entry_id}.A",
            "asym_id": "T",
            "auth_asym_id": "A",
            "uniprot_id": uniprot,
        }
    ]
    if "U" in neighbor_targets:
        mappings.append(
            {
                "entity_id": "3",
                "instance_id": f"{entry_id}.B",
                "asym_id": "U",
                "auth_asym_id": "B",
                "uniprot_id": "P22222",
            }
        )
    return {
        "schema_version": "skinscout.rcsb-holo-contact-snapshot.v1",
        "entry_id": entry_id,
        "initial_release_date": release,
        "experimental_methods": [{"method": "X-RAY DIFFRACTION"}],
        "resolution_combined": [2.0],
        "primary_citation": {"pdbx_database_id_DOI": f"10.1000/{entry_id.lower()}"},
        "human_polymer_uniprot_asym_mappings": mappings,
        "nonpolymer_entities": [
            {
                "rcsb_id": f"{entry_id}_2",
                "rcsb_nonpolymer_entity_container_identifiers": {
                    "nonpolymer_comp_id": comp_id,
                    "entity_id": "2",
                },
                "nonpolymer_comp": {
                    "chem_comp": {"id": comp_id, "formula": "C9 H8 O4", "formula_weight": 180.159},
                    "rcsb_chem_comp_descriptor": {
                        "SMILES": smiles,
                        "SMILES_stereo": smiles_stereo,
                        "InChI": inchi,
                        "InChIKey": inchikey,
                    },
                },
                "rcsb_nonpolymer_entity_annotation": [],
                "nonpolymer_entity_instances": [
                    {
                        "rcsb_id": f"{entry_id}.L",
                        "rcsb_nonpolymer_entity_instance_container_identifiers": {
                            "asym_id": "L",
                            "auth_asym_id": "L",
                            "comp_id": comp_id,
                            "entity_id": "2",
                        },
                        "rcsb_nonpolymer_instance_annotation": [],
                        "rcsb_target_neighbors": neighbors,
                        "rcsb_nonpolymer_struct_conn": [],
                    }
                ],
            }
        ],
        "subject_of_investigation_annotations": annotations,
        "rcsb_target_neighbors": [{"instance_id": f"{entry_id}.L", "neighbors": neighbors}],
        "rcsb_nonpolymer_struct_conn": [
            {
                "connect_type": "covale",
                "connect_partner": {"label_asym_id": "L"},
                "connect_target": {"label_asym_id": "T"},
            }
        ]
        if struct_conn
        else [],
    }


def _bench_row(**overrides: object) -> dict[str, object]:
    row = {
        "uniprot": "Q99999",
        "publication_key": "doi:10.1000/no-leak",
        "ligand_inchikey": "NOLEAK",
        "ligand_id": "L0",
        "ligand_smiles": "CCO",
        "scaffold_id": "s0",
        "target_cluster_30": "Q99999",
        "target_cluster_50": "Q99999",
    }
    row.update(overrides)
    return row


def _write_inputs(
    tmp_path: Path,
    rows: list[dict[str, object]],
    *,
    train_rows: list[dict[str, object]] | None = None,
    dev_rows: list[dict[str, object]] | None = None,
) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    raw = tmp_path / "raw.jsonl.gz"
    _write_gz_jsonl(raw, rows)
    source_manifest = tmp_path / "source_manifest.json"
    search_contract = {"q": "x"}
    graphql_contract = "query"
    source_manifest.write_text(
        json.dumps(
            {
                "schema_version": "skinscout.rcsb-holo-contact-snapshot.v1",
                "source_license": {"name": "CC0", "url": "https://www.rcsb.org/pages/policies"},
                "usage_contract": {
                    "positive_only_direct_contact_evaluation_source": True,
                    "never_training_or_calibration": True,
                },
                "release_cutoff": "2025-01-01",
                "api_query_contract": {
                    "search": search_contract,
                    "graphql": graphql_contract,
                },
                "query_sha256s": {
                    "search": _hash_json(search_contract),
                    "graphql": hashlib.sha256(graphql_contract.encode()).hexdigest(),
                },
                "candidate_count": len(rows),
                "candidate_record_counts": {"with_human_uniprot_asym_mapping": len(rows)},
                "filters": {
                    "experimental_entries": True,
                    "human_taxonomy_lineage_id": 9606,
                    "initial_release_date_gte": "2025-01-01",
                    "nonpolymer_molecular_weight_gt": 50.0,
                    "pdb_native_subject_of_investigation": True,
                },
                "raw_jsonl_gz": {
                    "path": str(raw.resolve()),
                    "sha256": _sha256(raw),
                    "bytes": raw.stat().st_size,
                    "rows": len(rows),
                },
            }
        )
        + "\n"
    )
    train = tmp_path / "train.parquet"
    dev = tmp_path / "dev.parquet"
    pd.DataFrame(train_rows or [_bench_row()]).to_parquet(train, index=False)
    pd.DataFrame(dev_rows or [_bench_row(uniprot="Q88888", target_cluster_30="Q88888", target_cluster_50="Q88888")]).to_parquet(dev, index=False)
    bench_manifest = tmp_path / "benchmark_manifest.json"
    bench_manifest.write_text(
        json.dumps(
            {
                "schema_version": "activity_benchmark.v1",
                "output_sha256": {"train.parquet": _sha256(train), "dev.parquet": _sha256(dev)},
                "splits": {"counts": {"train": len(pd.read_parquet(train)), "dev": len(pd.read_parquet(dev))}},
            }
        )
        + "\n"
    )
    clusters = tmp_path / "clusters.csv"
    pd.DataFrame(
        [
            {"uniprot": "P11111", "target_cluster_30": "COLD30", "target_cluster_50": "COLD50"},
            {"uniprot": "P22222", "target_cluster_30": "HOT30", "target_cluster_50": "HOT50"},
        ]
    ).to_csv(clusters, index=False)
    cluster_manifest = tmp_path / "clusters.manifest.json"
    cluster_manifest.write_text(
        json.dumps(
            {
                "schema_version": "skinscout.screenable-target-cluster-map.v2",
                "artifact": {"path": str(clusters.resolve()), "sha256": _sha256(clusters), "rows": 2},
                "production_contract": {"passes": True},
                "universe_policy": {
                    "evaluation_panel_used": False,
                    "known_target_assistance": False,
                    "union_target_count": 2,
                },
                "excluded_targets": {"count": 0, "records": []},
            }
        )
        + "\n"
    )
    return {
        "raw": raw,
        "source_manifest": source_manifest,
        "train": train,
        "dev": dev,
        "benchmark_manifest": bench_manifest,
        "clusters": clusters,
        "cluster_manifest": cluster_manifest,
    }


def _run(paths: dict[str, Path], out: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--raw-jsonl-gz",
            str(paths["raw"]),
            "--source-manifest",
            str(paths["source_manifest"]),
            "--train-parquet",
            str(paths["train"]),
            "--dev-parquet",
            str(paths["dev"]),
            "--benchmark-manifest",
            str(paths["benchmark_manifest"]),
            "--screenable-target-clusters",
            str(paths["clusters"]),
            "--screenable-target-cluster-manifest",
            str(paths["cluster_manifest"]),
            "--out-pairs",
            str(out / "pairs.parquet"),
            "--out-exclusions",
            str(out / "exclusions.csv"),
            "--out-ranking-queries",
            str(out / "ranking.parquet"),
            "--out-manifest",
            str(out / "manifest.json"),
            *extra,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_accepted_pair_ranking_manifest_and_model_compatibility(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path, [_raw_entry("1AAA")])
    result = _run(
        paths,
        tmp_path / "out",
        "--min-queries",
        "1",
        "--min-targets",
        "1",
        "--min-documents",
        "1",
        "--max-target-fraction",
        "1",
        "--min-effective-targets",
        "1",
    )
    assert result.returncode == 0, result.stderr

    pairs = pd.read_parquet(tmp_path / "out/pairs.parquet")
    ranking = pd.read_parquet(tmp_path / "out/ranking.parquet")
    manifest = json.loads((tmp_path / "out/manifest.json").read_text())

    assert len(pairs) == 1
    assert pairs.loc[0, "uniprot"] == "P11111"
    assert pairs.loc[0, "identity_route"].startswith("source_stereo_smiles_inchikey_match")
    assert ranking.loc[0, "truth_targets"] == ["P11111"]
    assert set(ranking[list(sorted({"claimable", "absent_scaffold_from_prior_splits", "absent_target_cluster_30_from_prior_splits", "absent_target_cluster_50_from_prior_splits"}))].iloc[0]) == {True}
    assert manifest["passes_adequacy"] is True
    assert manifest["contract"]["positive_only"] is True
    assert manifest["contract"]["pocket_leakage_audited"] is False
    assert "manifest" not in manifest["outputs"]
    for record in manifest["outputs"].values():
        assert _sha256(Path(record["path"])) == record["sha256"]

    sys.path.insert(0, str(ROOT / "eval"))
    from activity_retrieval_model import load_ranking_panel

    loaded = load_ranking_panel(tmp_path / "out/ranking.parquet", "test")
    assert loaded["query_id"].tolist() == ranking["query_id"].tolist()


def test_major_candidate_exclusions_are_recorded(tmp_path: Path) -> None:
    rows = [
        _raw_entry("1AAA", subject=False),
        _raw_entry("1AAB", neighbor_targets=["T", "U"]),
        _raw_entry("1AAC", residue_count=2),
        _raw_entry("1AAD", struct_conn=True),
        _raw_entry("1AAE", smiles="CCO", smiles_stereo="CCO", inchikey=Chem.MolToInchiKey(Chem.MolFromSmiles("CCO"))),
        _raw_entry("1AAF", uniprot="P33333"),
    ]
    paths = _write_inputs(tmp_path, rows)
    result = _run(paths, tmp_path / "out")
    assert result.returncode == 0, result.stderr
    exclusions = pd.read_csv(tmp_path / "out/exclusions.csv")
    assert set(exclusions["reason"]) == {
        "not_pdb_native_subject_of_investigation",
        "not_exactly_one_contacted_human_target",
        "insufficient_unique_contact_residues",
        "covalent_or_metal_coordination_struct_conn",
        "ligand_identity_or_chemistry_failed",
        "target_not_screenable",
    }


def test_inchi_fallback_accepts_when_smiles_key_mismatches(tmp_path: Path) -> None:
    aspirin_smiles, aspirin_key, aspirin_inchi = _aspirin()
    paths = _write_inputs(
        tmp_path,
        [_raw_entry("1AAA", smiles="CCO", smiles_stereo="CCO", inchi=aspirin_inchi, inchikey=aspirin_key)],
    )
    result = _run(paths, tmp_path / "out", "--min-queries", "1", "--min-targets", "1", "--min-documents", "1", "--max-target-fraction", "1", "--min-effective-targets", "1")
    assert result.returncode == 0, result.stderr
    pairs = pd.read_parquet(tmp_path / "out/pairs.parquet")
    assert len(pairs) == 1
    assert pairs.loc[0, "identity_route"].startswith("source_inchi_inchikey_match")
    assert pairs.loc[0, "source_identity_smiles"] == Chem.MolToSmiles(Chem.MolFromSmiles(aspirin_smiles), canonical=True, isomericSmiles=True)


def test_blank_source_inchikey_is_explicitly_excluded(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path, [_raw_entry("1AAA", inchikey="")])
    result = _run(paths, tmp_path / "out")
    assert result.returncode == 0, result.stderr
    exclusions = pd.read_csv(tmp_path / "out/exclusions.csv")
    assert exclusions["reason"].tolist() == ["ligand_identity_or_chemistry_failed"]
    assert exclusions["detail"].tolist() == ["missing_source_inchikey"]
    assert pd.read_parquet(tmp_path / "out/pairs.parquet").empty


def test_train_only_leakage_sets_train_and_prior_flags_independently(tmp_path: Path) -> None:
    _smiles, aspirin_key, _inchi = _aspirin()
    paths = _write_inputs(
        tmp_path,
        [_raw_entry("1AAA")],
        train_rows=[
            _bench_row(
                uniprot="P11111",
                publication_key="doi:10.1000/1aaa",
                ligand_inchikey=aspirin_key,
                scaffold_id="bemis_murcko:" + hashlib.sha256("c1ccccc1".encode()).hexdigest(),
                target_cluster_30="COLD30",
                target_cluster_50="COLD50",
            )
        ],
    )
    result = _run(paths, tmp_path / "out")
    assert result.returncode == 0, result.stderr
    pairs = pd.read_parquet(tmp_path / "out/pairs.parquet")
    ranking = pd.read_parquet(tmp_path / "out/ranking.parquet")
    assert len(pairs) == 1
    assert bool(pairs.loc[0, "absent_pair_from_train"]) is False
    assert bool(pairs.loc[0, "absent_pair_from_prior_splits"]) is False
    assert bool(pairs.loc[0, "absent_publication_from_train"]) is False
    assert bool(pairs.loc[0, "absent_publication_from_prior_splits"]) is False
    assert bool(pairs.loc[0, "absent_scaffold_from_train"]) is False
    assert bool(pairs.loc[0, "absent_scaffold_from_prior_splits"]) is False
    assert bool(pairs.loc[0, "absent_target_cluster_30_from_train"]) is False
    assert bool(pairs.loc[0, "absent_target_cluster_30_from_prior_splits"]) is False
    assert ranking.empty


def test_dev_only_leakage_keeps_train_flags_true_and_prior_flags_false(tmp_path: Path) -> None:
    paths = _write_inputs(
        tmp_path,
        [_raw_entry("1AAA")],
        dev_rows=[
            _bench_row(
                uniprot="P99999",
                publication_key="doi:10.1000/1aaa",
                target_cluster_30="COLD30",
                target_cluster_50="COLD50",
            )
        ],
    )
    result = _run(paths, tmp_path / "out")
    assert result.returncode == 0, result.stderr
    pairs = pd.read_parquet(tmp_path / "out/pairs.parquet")
    assert len(pairs) == 1
    assert bool(pairs.loc[0, "absent_publication_from_train"]) is True
    assert bool(pairs.loc[0, "absent_publication_from_prior_splits"]) is False
    assert bool(pairs.loc[0, "absent_target_cluster_30_from_train"]) is True
    assert bool(pairs.loc[0, "absent_target_cluster_30_from_prior_splits"]) is False
    assert bool(pairs.loc[0, "is_dual_cold"]) is False


def test_deduplicates_earliest_structure_and_merges_targets_per_query(tmp_path: Path) -> None:
    paths = _write_inputs(
        tmp_path,
        [
            _raw_entry("1AAA", release="2025-03-01"),
            _raw_entry("1AAB", release="2025-02-01"),
            _raw_entry("1AAC", uniprot="P22222", release="2025-04-01"),
        ],
    )
    result = _run(paths, tmp_path / "out", "--min-queries", "1", "--min-targets", "1", "--min-documents", "1", "--max-target-fraction", "1", "--min-effective-targets", "1")
    assert result.returncode == 0, result.stderr
    pairs = pd.read_parquet(tmp_path / "out/pairs.parquet")
    ranking = pd.read_parquet(tmp_path / "out/ranking.parquet")
    assert pairs[pairs["uniprot"].eq("P11111")]["entry_id"].tolist() == ["1AAB"]
    assert list(ranking.loc[0, "truth_targets"]) == ["P11111", "P22222"]


def test_manifest_tampering_raw_corruption_and_cluster_provenance_mismatch_fail_closed(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path / "tamper", [_raw_entry("1AAA")])
    payload = json.loads(paths["source_manifest"].read_text())
    payload["raw_jsonl_gz"]["sha256"] = "0" * 64
    paths["source_manifest"].write_text(json.dumps(payload) + "\n")
    result = _run(paths, tmp_path / "tamper/out")
    assert result.returncode != 0
    assert "sha256" in result.stderr

    corrupt = _write_inputs(tmp_path / "corrupt", [_raw_entry("1AAA")])
    corrupt["raw"].write_bytes(b"not gzip")
    result = _run(corrupt, tmp_path / "corrupt/out")
    assert result.returncode != 0
    assert "sha256" in result.stderr or "gzipped raw" in result.stderr

    mismatch = _write_inputs(tmp_path / "cluster", [_raw_entry("1AAA")])
    cluster_manifest = json.loads(mismatch["cluster_manifest"].read_text())
    cluster_manifest["artifact"]["rows"] = 99
    mismatch["cluster_manifest"].write_text(json.dumps(cluster_manifest) + "\n")
    result = _run(mismatch, tmp_path / "cluster/out")
    assert result.returncode != 0
    assert "row count" in result.stderr


def test_benchmark_publication_key_and_distinct_split_paths_are_required(tmp_path: Path) -> None:
    missing_publication = _write_inputs(tmp_path / "missing-publication", [_raw_entry("1AAA")])
    pd.DataFrame([_bench_row()]).drop(columns=["publication_key"]).to_parquet(
        missing_publication["train"], index=False
    )
    payload = json.loads(missing_publication["benchmark_manifest"].read_text())
    payload["output_sha256"]["train.parquet"] = _sha256(missing_publication["train"])
    missing_publication["benchmark_manifest"].write_text(json.dumps(payload) + "\n")
    result = _run(missing_publication, tmp_path / "missing-publication/out")
    assert result.returncode != 0
    assert "publication_key" in result.stderr

    same_path = _write_inputs(tmp_path / "same-path", [_raw_entry("1AAA")])
    same_path["dev"] = same_path["train"]
    result = _run(same_path, tmp_path / "same-path/out")
    assert result.returncode != 0
    assert "distinct" in result.stderr


def test_deterministic_output_and_underpowered_gate(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path, [_raw_entry("1AAA")])
    first = _run(paths, tmp_path / "out")
    assert first.returncode == 0, first.stderr
    first_pairs = json.loads(
        pd.read_parquet(tmp_path / "out/pairs.parquet").to_json(orient="records")
    )
    first_hash = _sha256(tmp_path / "out/ranking.parquet")

    second = _run(paths, tmp_path / "out2")
    assert second.returncode == 0, second.stderr
    second_pairs = json.loads(
        pd.read_parquet(tmp_path / "out2/pairs.parquet").to_json(orient="records")
    )
    assert first_pairs == second_pairs
    assert first_hash == _sha256(tmp_path / "out2/ranking.parquet")

    manifest = json.loads((tmp_path / "out/manifest.json").read_text())
    assert manifest["passes_adequacy"] is False
    assert manifest["selection"]["ranking_queries"]["test"]["adequacy"]["passes"] is False
