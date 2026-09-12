from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/build_activity_retrieval_index.py"
sys.path.insert(0, str(ROOT / "scripts"))

from stage3_daina_zoete import fp_from_uint64_row  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _base_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "split": "train",
        "source_db": "chembl",
        "uniprot": "P12345",
        "ligand_smiles": "CCO",
        "publication_key": "pmid:1",
        "endpoint": "IC50",
        "pactivity": 6.0,
    }
    row.update(overrides)
    return row


def _write_train_and_manifest(tmp_path: Path, rows: list[dict[str, object]]) -> tuple[Path, Path]:
    train = tmp_path / "train.parquet"
    pd.DataFrame(rows).to_parquet(train, index=False)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "activity_benchmark.v1",
                "output_sha256": {"train.parquet": _sha256(train)},
                "splits": {"counts": {"train": len(rows)}},
            }
        )
        + "\n"
    )
    return train, manifest


def _run_builder(
    tmp_path: Path,
    train: Path,
    manifest: Path,
    *extra: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--train-parquet",
            str(train),
            "--benchmark-manifest",
            str(manifest),
            "--out-ligands",
            str(tmp_path / "ligands.parquet"),
            "--out-edges",
            str(tmp_path / "edges.parquet"),
            "--out-manifest",
            str(tmp_path / "index_manifest.json"),
            *extra,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_provenance_hash_mismatch_fails_closed_and_removes_stale_outputs(
    tmp_path: Path,
) -> None:
    train, manifest = _write_train_and_manifest(tmp_path, [_base_row()])
    payload = json.loads(manifest.read_text())
    payload["output_sha256"]["train.parquet"] = "0" * 64
    manifest.write_text(json.dumps(payload) + "\n")
    for name in ["ligands.parquet", "edges.parquet", "index_manifest.json"]:
        (tmp_path / name).write_text("stale\n")

    res = _run_builder(tmp_path, train, manifest)

    assert res.returncode != 0
    assert "sha256 does not match" in res.stderr
    assert not (tmp_path / "ligands.parquet").exists()
    assert not (tmp_path / "edges.parquet").exists()
    assert not (tmp_path / "index_manifest.json").exists()


def test_deterministic_aggregation_bit_roundtrip_source_separation_and_cx_fallback(
    tmp_path: Path,
) -> None:
    cx_smiles = "CCO |badCX:1|"
    rows = [
        _base_row(ligand_smiles=cx_smiles, source_db="chembl", pactivity=6.0),
        _base_row(ligand_smiles="CCO", source_db="chembl", pactivity=5.5),
        _base_row(ligand_smiles="CCO", source_db="chembl", pactivity=4.0),
        _base_row(ligand_smiles="CCO", source_db="chembl", endpoint="Ki", pactivity=7.0),
        _base_row(ligand_smiles="CCO", source_db="bindingdb", pactivity=7.0),
        _base_row(ligand_smiles="c1ccccc1O", publication_key="pmid:2", pactivity=5.0),
    ]
    train, manifest = _write_train_and_manifest(tmp_path, rows)

    first = _run_builder(tmp_path, train, manifest)
    assert first.returncode == 0, first.stderr
    first_ligand_hash = _sha256(tmp_path / "ligands.parquet")
    first_edge_hash = _sha256(tmp_path / "edges.parquet")

    second = _run_builder(tmp_path, train, manifest)
    assert second.returncode == 0, second.stderr
    assert _sha256(tmp_path / "ligands.parquet") == first_ligand_hash
    assert _sha256(tmp_path / "edges.parquet") == first_edge_hash

    ligands = pd.read_parquet(tmp_path / "ligands.parquet")
    edges = pd.read_parquet(tmp_path / "edges.parquet")
    assert list(ligands.columns) == [
        "ligand_index",
        "ligand_key",
        "standard_inchikey",
        "connectivity_key",
        "canonical_smiles",
        "standardization_route",
        "bitvec",
    ]
    assert ligands["ligand_index"].tolist() == list(range(len(ligands)))
    assert ligands["ligand_key"].tolist() == sorted(ligands["ligand_key"].tolist())
    ethanol = ligands[ligands["canonical_smiles"] == "CCO"].iloc[0]
    expected = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
        includeChirality=False,
    ).GetFingerprint(Chem.MolFromSmiles("CCO"))
    restored = fp_from_uint64_row(list(ethanol["bitvec"]))
    assert restored.ToBitString() == expected.ToBitString()
    assert DataStructs.TanimotoSimilarity(restored, expected) == 1.0

    ethanol_edges = edges[edges["ligand_index"] == int(ethanol["ligand_index"])]
    assert sorted(
        zip(ethanol_edges["source_db"], ethanol_edges["endpoint_family"], strict=True)
    ) == [
        ("bindingdb", "functional"),
        ("chembl", "direct_binding"),
        ("chembl", "functional"),
    ]
    chembl = ethanol_edges[
        (ethanol_edges["source_db"] == "chembl")
        & (ethanol_edges["endpoint_family"] == "functional")
    ].iloc[0]
    assert int(chembl["measurement_count"]) == 3
    assert int(chembl["positive_measurement_count"]) == 1
    assert int(chembl["gray_measurement_count"]) == 1
    assert int(chembl["negative_measurement_count"]) == 1
    assert float(chembl["max_pactivity"]) == 6.0
    assert float(chembl["median_pactivity"]) == 5.5
    assert int(chembl["endpoint_count"]) == 1
    assert int(chembl["publication_count"]) == 1

    index_manifest = json.loads((tmp_path / "index_manifest.json").read_text())
    assert index_manifest["schema_version"] == "skinscout.activity-retrieval-index.v4"
    assert Path(index_manifest["inputs"]["train_parquet"]["path"]).is_absolute()
    assert Path(index_manifest["outputs"]["ligands"]["path"]).is_absolute()
    assert index_manifest["algorithm"]["label_policy"]["negative_threshold"] == 5.0
    assert index_manifest["algorithm"]["label_policy"]["positive_threshold"] == 6.0
    assert index_manifest["algorithm"]["label_policy"]["unmeasured_pairs"].startswith(
        "unlabeled"
    )
    assert index_manifest["source_counts"] == {"bindingdb": 1, "chembl": 5}


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (_base_row(ligand_smiles="not a smiles"), "invalid ligand_smiles"),
        # A dev row still fails the default build; the message now names the
        # permitted set, because a production index may declare more than train.
        (_base_row(split="dev"), "split in ['train']"),
        (_base_row(pactivity=float("nan")), "non-finite pactivity"),
        (_base_row(source_db=" "), "blank source_db"),
        (_base_row(endpoint="MIC"), "unsupported endpoint"),
    ],
)
def test_invalid_nontrain_and_nonfinite_inputs_fail_closed(
    tmp_path: Path,
    row: dict[str, object],
    message: str,
) -> None:
    train, manifest = _write_train_and_manifest(tmp_path, [row])

    res = _run_builder(tmp_path, train, manifest)

    assert res.returncode != 0
    assert message in res.stderr
    assert not (tmp_path / "ligands.parquet").exists()
    assert not (tmp_path / "edges.parquet").exists()
    assert not (tmp_path / "index_manifest.json").exists()


def test_label_thresholds_must_define_a_nonempty_gray_zone(tmp_path: Path) -> None:
    train, manifest = _write_train_and_manifest(tmp_path, [_base_row()])

    res = _run_builder(
        tmp_path,
        train,
        manifest,
        "--negative-threshold",
        "6",
        "--positive-threshold",
        "6",
    )

    assert res.returncode != 0
    assert "must be less than" in res.stderr


def test_parallel_workers_preserve_serial_output(tmp_path: Path) -> None:
    rows = [
        _base_row(ligand_smiles="CCO", pactivity=6.0),
        _base_row(ligand_smiles="c1ccccc1O", publication_key="pmid:2", pactivity=5.0),
        _base_row(
            ligand_smiles="CC(=O)O.[Na+]",
            publication_key="pmid:3",
            pactivity=4.0,
        ),
    ]
    train, manifest = _write_train_and_manifest(tmp_path, rows)

    serial = _run_builder(tmp_path, train, manifest)
    assert serial.returncode == 0, serial.stderr
    serial_ligands = pd.read_parquet(tmp_path / "ligands.parquet")
    serial_edges = pd.read_parquet(tmp_path / "edges.parquet")

    parallel = _run_builder(tmp_path, train, manifest, "--workers", "2")
    assert parallel.returncode == 0, parallel.stderr
    pd.testing.assert_frame_equal(
        pd.read_parquet(tmp_path / "ligands.parquet"),
        serial_ligands,
    )
    pd.testing.assert_frame_equal(
        pd.read_parquet(tmp_path / "edges.parquet"),
        serial_edges,
    )


def test_worker_count_must_be_positive(tmp_path: Path) -> None:
    train, manifest = _write_train_and_manifest(tmp_path, [_base_row()])

    res = _run_builder(tmp_path, train, manifest, "--workers", "0")

    assert res.returncode != 0
    assert "--workers must be positive" in res.stderr


def test_progress_interval_must_be_nonnegative(tmp_path: Path) -> None:
    train, manifest = _write_train_and_manifest(tmp_path, [_base_row()])

    res = _run_builder(tmp_path, train, manifest, "--progress-every", "-1")

    assert res.returncode != 0
    assert "--progress-every must be non-negative" in res.stderr


def test_standard_inchi_tautomers_remain_explicit_structure_variants(tmp_path: Path) -> None:
    rows = [
        _base_row(ligand_smiles="Brc1ccc(-c2nnc(-c3ccccc3)[nH]2)cc1"),
        _base_row(
            ligand_smiles="Brc1ccc(-c2n[nH]c(-c3ccccc3)n2)cc1",
            source_db="bindingdb",
        ),
    ]
    train, manifest = _write_train_and_manifest(tmp_path, rows)

    res = _run_builder(tmp_path, train, manifest)

    assert res.returncode == 0, res.stderr
    ligands = pd.read_parquet(tmp_path / "ligands.parquet")
    edges = pd.read_parquet(tmp_path / "edges.parquet")
    assert len(ligands) == 2
    assert set(ligands["standard_inchikey"]) == {"PLRURNUSAXSWDI-UHFFFAOYSA-N"}
    assert ligands["ligand_key"].str.startswith(
        "PLRURNUSAXSWDI-UHFFFAOYSA-N#SMILES-"
    ).all()
    assert set(ligands["standardization_route"]) == {
        "fragment_parent_canonical_smiles_key"
    }
    assert len(edges) == 2


def test_structure_keys_are_dataset_independent_for_singletons(tmp_path: Path) -> None:
    train, manifest = _write_train_and_manifest(tmp_path, [_base_row(ligand_smiles="CCO")])

    result = _run_builder(tmp_path, train, manifest)

    assert result.returncode == 0, result.stderr
    ligand = pd.read_parquet(tmp_path / "ligands.parquet").iloc[0]
    assert ligand["ligand_key"].startswith(
        "LFQSCWFLJHTTHZ-UHFFFAOYSA-N#SMILES-"
    )
    assert ligand["standardization_route"] == (
        "fragment_parent_canonical_smiles_key"
    )


@pytest.mark.parametrize(
    ("charged", "neutral"),
    [
        ("CC(=O)[O-]", "CC(=O)O"),
        ("[NH3+]CC(=O)[O-]", "NCC(=O)O"),
    ],
)
def test_charged_and_neutral_query_forms_share_exact_index_identity(
    charged: str, neutral: str
) -> None:
    from activity_retrieval_scoring import query_features
    from build_activity_retrieval_index import _standardize_ligand

    index_key, _, canonical, _ = _standardize_ligand(charged)
    query_fp, query_key, _, query_canonical = query_features(neutral)
    charged_fp, charged_key, _, charged_canonical = query_features(charged)

    assert charged_key == query_key == index_key
    assert charged_canonical == query_canonical == canonical
    assert DataStructs.TanimotoSimilarity(charged_fp, query_fp) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "smiles",
    [
        "C[P-](=O)(=O)c1cn(-c2ccc3ncnc(Nc4c(F)ccc(NS(=O)(=O)c5cccc(Cl)c5Cl)c4F)c3n2)c2cccc(N)c12",
        "C[C@H]1C2NN(C)C(c3cc(F)cc(C[P-](C)(=O)=O)c3)=C2CCN1C(=O)c1cc(F)cc(C2C=C(C#N)NC2)c1Cl",
        "C[C@@H]1C2NN(C)C(c3cc(F)cc(C[P-](C)(=O)=O)c3)=C2CCN1C(=O)c1cc(F)cc(C2C=C(C#N)NC2)c1Cl",
    ],
)
def test_anionic_phosphinates_keep_a_valid_structure_on_both_sides(smiles: str) -> None:
    """Uncharger output for anionic phosphinates does not re-parse (audit H-11).

    The index used to fail these three and the run path kept them, so the two
    sides could disagree about a compound's own structure. Both now go through
    the charged-parent fallback in `standardize_parent`.
    """
    from activity_retrieval_scoring import query_features
    from build_activity_retrieval_index import MORGAN_GENERATOR, _standardize_ligand

    index_key, _, canonical, _ = _standardize_ligand(smiles)
    query_fp, query_key, _, query_canonical = query_features(smiles)

    assert Chem.MolFromSmiles(canonical) is not None
    assert query_key == index_key
    assert query_canonical == canonical
    assert DataStructs.TanimotoSimilarity(
        MORGAN_GENERATOR.GetFingerprint(Chem.MolFromSmiles(canonical)), query_fp
    ) == pytest.approx(1.0)


def test_kekule_fallback_retains_fragment_parent_that_cannot_reparse_as_aromatic_smiles(
    tmp_path: Path,
) -> None:
    raw = "CCOC(=O)[N-]c1cn(n[o+]1)N1CCOCC1"
    train, manifest = _write_train_and_manifest(
        tmp_path, [_base_row(ligand_smiles=raw)]
    )

    result = _run_builder(tmp_path, train, manifest)

    assert result.returncode == 0, result.stderr
    ligand = pd.read_parquet(tmp_path / "ligands.parquet").iloc[0]
    assert ligand["canonical_smiles"] == "CCOC(=O)N=C1CN(N2CCOCC2)NO1"
    assert ligand["standard_inchikey"] == "IUVFAERPPHEQSB-UHFFFAOYSA-N"
    assert Chem.MolFromSmiles(ligand["canonical_smiles"]) is not None
