"""A production index may read dev and test; the evaluation gate must refuse it.

The retrieval index is built from the benchmark's train split (<=2023-12-31).
That is right for measuring recovery - it is the whole basis of the temporal
split - but it is not a constraint a researcher's own run needs, and it costs
133 real targets, including AQP3 (Q92482), the keratinocyte water/glycerol
channel a moisturising ingredient should surface.

So the builder can now produce a second index that also reads dev and test.
The danger is obvious: if evaluation ever consumed that index it would be
scoring the model on rows it retrieves from, and every recovery number in the
repository would quietly become optimistic. These tests pin the separation.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build_activity_retrieval_index.py"
GATE = ROOT / "scripts" / "validate_activity_retrieval_gate.py"
RUNTIME_BUILDER = ROOT / "scripts" / "build_runtime_retrieval_index.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _row(split: str, uniprot: str, smiles: str, pub: str) -> dict[str, object]:
    return {
        "split": split,
        "source_db": "chembl",
        "uniprot": uniprot,
        "ligand_smiles": smiles,
        "publication_key": pub,
        "endpoint": "IC50",
        "pactivity": 6.5,
    }


@pytest.fixture
def benchmark(tmp_path: Path) -> dict[str, Path]:
    """train has one target; dev and test each bring one the train split lacks."""
    splits = {
        "train": [_row("train", "P11111", "CCO", "pmid:1")],
        "dev": [_row("dev", "P22222", "CCCO", "pmid:2")],
        "test": [_row("test", "Q92482", "CCCCO", "pmid:3")],
    }
    paths = {}
    for name, rows in splits.items():
        path = tmp_path / f"{name}.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        paths[name] = path

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "activity_benchmark.v1",
                "output_sha256": {
                    f"{name}.parquet": _sha256(path) for name, path in paths.items()
                },
                "splits": {"counts": {name: len(rows) for name, rows in splits.items()}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    paths["manifest"] = manifest
    return paths


def _build(tmp_path: Path, benchmark: dict[str, Path], *extra: str, out: str = "out"):
    directory = tmp_path / out
    directory.mkdir(exist_ok=True)
    return (
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--train-parquet",
                str(benchmark["train"]),
                "--benchmark-manifest",
                str(benchmark["manifest"]),
                "--out-ligands",
                str(directory / "ligands.parquet"),
                "--out-edges",
                str(directory / "edges.parquet"),
                "--out-manifest",
                str(directory / "manifest.json"),
                *extra,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        ),
        directory,
    )


def test_the_default_build_is_still_train_only(tmp_path: Path, benchmark) -> None:
    result, directory = _build(tmp_path, benchmark)

    assert result.returncode == 0, result.stderr
    payload = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "skinscout.activity-retrieval-index.v4"
    assert payload["index_role"] == "evaluation"
    assert set(payload["inputs"]["splits"]) == {"train"}

    edges = pd.read_parquet(directory / "edges.parquet")
    assert set(edges["uniprot"]) == {"P11111"}


def test_extra_splits_without_the_production_role_are_refused(tmp_path: Path, benchmark) -> None:
    """The role has to be chosen, never inferred from the flags that were passed."""
    result, _ = _build(tmp_path, benchmark, "--dev-parquet", str(benchmark["dev"]))

    assert result.returncode != 0
    assert "--index-role production" in result.stderr


def test_the_production_role_without_extra_splits_is_refused(tmp_path: Path, benchmark) -> None:
    """Otherwise it is the evaluation index wearing a name that says otherwise."""
    result, _ = _build(tmp_path, benchmark, "--index-role", "production")

    assert result.returncode != 0
    assert "at least one of" in result.stderr


def test_a_production_index_reaches_the_targets_train_alone_cannot(
    tmp_path: Path, benchmark
) -> None:
    result, directory = _build(
        tmp_path,
        benchmark,
        "--index-role",
        "production",
        "--dev-parquet",
        str(benchmark["dev"]),
        "--test-parquet",
        str(benchmark["test"]),
    )

    assert result.returncode == 0, result.stderr
    edges = pd.read_parquet(directory / "edges.parquet")
    # Q92482 is AQP3's accession; in the real data it lives in the test split.
    assert set(edges["uniprot"]) == {"P11111", "P22222", "Q92482"}

    payload = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "skinscout.activity-retrieval-index.production.v1"
    assert payload["index_role"] == "production"
    assert set(payload["inputs"]["splits"]) == {"train", "dev", "test"}
    for split in ("train", "dev", "test"):
        assert payload["inputs"]["splits"][split]["sha256"] == _sha256(benchmark[split])


def test_a_tampered_extra_split_fails_closed(tmp_path: Path, benchmark) -> None:
    """dev and test get the same provenance binding train has, not a weaker one."""
    pd.DataFrame([_row("dev", "P33333", "CCCCCO", "pmid:9")]).to_parquet(
        benchmark["dev"], index=False
    )

    result, _ = _build(
        tmp_path,
        benchmark,
        "--index-role",
        "production",
        "--dev-parquet",
        str(benchmark["dev"]),
    )

    assert result.returncode != 0
    assert "sha256 does not match" in result.stderr


def _malformed_benchmark(tmp_path: Path) -> dict[str, Path]:
    """One row carries the SMILES that stopped the first production build.

    `C:C` is an aromatic bond between two acyclic carbons. RDKit parses it with a
    warning and then cannot generate an InChI, so the ligand has no key.
    """
    bad = "C:CC(=O)Nc1ccc2c(c1)c(=O)[nH]c1c(C(N)=O)c(-c3ccc(Oc4ccccc4)cc3)nn12"
    rows = [
        _row("train", "P11111", "CCO", "pmid:1"),
        _row("train", "P11111", bad, "pmid:2"),
    ]
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
        + "\n",
        encoding="utf-8",
    )
    return {"train": train, "manifest": manifest}


def test_one_unstandardizable_ligand_still_stops_the_build_by_default(
    tmp_path: Path,
) -> None:
    """The evaluation index keeps failing closed; the allowance defaults to zero."""
    benchmark = _malformed_benchmark(tmp_path)
    result, _ = _build(tmp_path, benchmark)

    assert result.returncode != 0
    assert "could not be standardized" in result.stderr
    assert "allowance of 0" in result.stderr


def test_a_declared_allowance_drops_the_ligand_and_names_it(tmp_path: Path) -> None:
    """A malformed record among a million should not block a rebuild - but the
    build has to say exactly what it threw away, in the manifest, not the log."""
    benchmark = _malformed_benchmark(tmp_path)
    result, directory = _build(tmp_path, benchmark, "--max-unusable-ligands", "1")

    assert result.returncode == 0, result.stderr
    payload = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert payload["unusable_ligands"]["count"] == 1
    record = payload["unusable_ligands"]["records"][0]
    assert record["ligand_smiles"].startswith("C:CC(=O)N")
    assert "InChIKey" in record["error"]

    # The good ligand still made it, and the bad row produced no edge.
    ligands = pd.read_parquet(directory / "ligands.parquet")
    edges = pd.read_parquet(directory / "edges.parquet")
    assert len(ligands) == 1
    assert len(edges) == 1


def test_a_clean_build_records_an_empty_exclusion_list(tmp_path: Path, benchmark) -> None:
    """Absence of the field would be indistinguishable from 'nobody checked'."""
    result, directory = _build(tmp_path, benchmark)

    assert result.returncode == 0, result.stderr
    payload = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert payload["unusable_ligands"] == {"count": 0, "records": []}


def test_the_evaluation_gate_refuses_a_production_index() -> None:
    """The one mistake that would silently inflate every recovery number."""
    source = GATE.read_text(encoding="utf-8")
    assert 'index.get("index_role", "evaluation")' in source
    assert "requires a train-only index" in source


def test_the_builder_documents_why_the_roles_are_separate() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "PRODUCTION_SCHEMA_VERSION" in source
    # The reason has to sit next to the code, not only in a commit message.
    assert "temporal split" in source


def test_a_merged_tables_own_publication_key_is_preferred(tmp_path: Path) -> None:
    """A merged evidence table carries `publication_key` because BindingDB's
    patent and article identifiers have no ChEMBL column to live in. Ignoring it
    dropped every such row at build; overwriting a ChEMBL row's key with it would
    be just as wrong, so it only fills where it is non-blank."""
    import pandas as pd

    sys.path.insert(0, str(ROOT / "scripts"))

    from build_runtime_retrieval_index import _publication_key

    frame = pd.DataFrame(
        {
            "publication_key": ["patent:US7538232", "", None],
            "pubmed_id": ["", "21334791", "111"],
            "doi": ["", "", ""],
            "document_chembl_id": ["", "", ""],
        }
    )
    assert list(_publication_key(frame)) == ["patent:US7538232", "pmid:21334791", "pmid:111"]


def test_a_table_without_the_merged_column_is_unaffected(tmp_path: Path) -> None:
    """ChEMBL-only evidence has no such column; the ChEMBL fields must still
    produce exactly the keys they did before the merged path existed."""
    import pandas as pd

    sys.path.insert(0, str(ROOT / "scripts"))

    from build_runtime_retrieval_index import _publication_key

    frame = pd.DataFrame(
        {"pubmed_id": ["21334791", ""], "doi": ["", "10.1/x"], "document_chembl_id": ["", "CHEMBL9"]}
    )
    assert list(_publication_key(frame)) == ["pmid:21334791", "doi:10.1/x"]


def _projection(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_the_index_records_the_licence_tier_it_was_built_from(tmp_path: Path) -> None:
    """A merged table can pull in share-alike sources, and once the index is
    built the evidence directory is a path away. The obligation has to travel
    with the index or nothing on the run path records it."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_runtime_retrieval_index import _evidence_licensing

    evidence = tmp_path / "merged"
    evidence.mkdir()
    (evidence / "manifest.json").write_text(
        json.dumps(
            {
                "sources_tier": "permissive",
                "sources": ["bindingdb_own"],
                "licences": ["CC BY 4.0 (BindingDB-curated)"],
            }
        ),
        encoding="utf-8",
    )
    record = _evidence_licensing(
        evidence,
        _projection(
            {
                "source_db": "BindingDB",
                "source_release": "2026-08",
                "source_license": "CC BY 4.0 (BindingDB-curated)",
            }
        ),
    )
    assert record["sources_tier"] == "permissive"
    assert record["licences"] == ["CC BY 4.0 (BindingDB-curated)"]
    assert record["source_metadata"]["BindingDB"]["rows"] == 1


def test_a_plain_chembl_mirror_still_declares_a_licence(tmp_path: Path) -> None:
    """`data/chembl37` has no merge manifest. Returning nothing would read as
    'no licence obligation' rather than 'the ChEMBL mirror as shipped'."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_runtime_retrieval_index import _evidence_licensing

    (tmp_path / "source_manifest.json").write_text(
        json.dumps({"row_counts": {"human_activities": 1}}), encoding="utf-8"
    )
    record = _evidence_licensing(
        tmp_path,
        _projection(
            {
                "source_db": "ChEMBL",
                "source_release": "37",
                "source_license": "CC BY-SA 3.0",
            }
        ),
    )
    assert record["sources_tier"] == "chembl-only"
    assert record["licences"] == ["CC BY-SA 3.0"]
    assert record["source_metadata"]["ChEMBL"]["releases"] == ["37"]
    assert Path(record["source_metadata"]["ChEMBL"]["source_manifest"]["path"]) == (
        tmp_path / "source_manifest.json"
    ).resolve()


def test_every_actual_source_gets_its_own_licence_record(tmp_path: Path) -> None:
    """The F24 bug: only the added tier's union licence travelled with the index,
    so the ChEMBL majority looked unlicensed. The per-source record must name
    each source that has projected rows."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_runtime_retrieval_index import _evidence_licensing

    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "sources_tier": "permissive",
                "sources": ["bindingdb_own"],
                "licences": ["CC BY 4.0 (BindingDB-curated)"],
            }
        ),
        encoding="utf-8",
    )
    record = _evidence_licensing(
        tmp_path,
        _projection(
            {
                "source_db": "ChEMBL",
                "source_release": "37",
                "source_license": "CC BY-SA 3.0",
            },
            {
                "source_db": "BindingDB",
                "source_release": "2026-08",
                "source_license": "CC BY 4.0 (BindingDB-curated)",
            },
        ),
    )
    assert set(record["source_metadata"]) == {"ChEMBL", "BindingDB"}
    assert record["licences"] == [
        "CC BY 4.0 (BindingDB-curated)",
        "CC BY-SA 3.0",
    ]
    assert record["source_metadata"]["ChEMBL"]["rows"] == 1


def test_a_source_without_release_metadata_stops_the_build(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_runtime_retrieval_index import _evidence_licensing

    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="has no release metadata"):
        _evidence_licensing(
            tmp_path,
            _projection(
                {
                    "source_db": "ChEMBL",
                    "source_release": "",
                    "source_license": "CC BY-SA 3.0",
                }
            ),
        )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--negative-threshold", "nan"], "must be finite"),
        (["--positive-threshold", "inf"], "must be finite"),
        (["--negative-threshold", "7", "--positive-threshold", "6"], "must be less"),
        (["--workers", "0"], "workers must be positive"),
        (["--max-unusable-ligands", "-1"], "must be non-negative"),
    ],
)
def test_runtime_builder_rejects_invalid_parameters_before_reading_inputs(
    tmp_path: Path, arguments: list[str], message: str
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(RUNTIME_BUILDER),
            "--evidence-dir",
            str(tmp_path / "missing"),
            "--out-dir",
            str(tmp_path / "out"),
            *arguments,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert message in result.stderr


def _runtime_evidence(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    directory = tmp_path / "evidence"
    directory.mkdir()
    pd.DataFrame(rows).to_parquet(
        directory / "human_activities.parquet", index=False
    )
    (directory / "source_manifest.json").write_text(
        json.dumps({"row_counts": {"human_activities": len(rows)}}), encoding="utf-8"
    )
    return directory


def _runtime_row(source: str, uniprot: str, smiles: str, pub: str) -> dict[str, object]:
    return {
        "act_type": "IC50",
        "pchembl": 6.5,
        "source_db": source,
        "source_release": "37" if source == "ChEMBL" else "2026-08",
        "source_license": (
            "CC BY-SA 3.0" if source == "ChEMBL" else "CC BY 4.0 (BindingDB-curated)"
        ),
        "uniprot": uniprot,
        "smiles": smiles,
        "pubmed_id": pub,
    }


def test_the_runtime_builder_records_every_actual_source(tmp_path: Path) -> None:
    evidence = _runtime_evidence(
        tmp_path,
        [
            _runtime_row("ChEMBL", "P11111", "CCO", "1"),
            _runtime_row("BindingDB", "P22222", "CCCO", "2"),
        ],
    )
    targets = tmp_path / "targets.csv"
    targets.write_text("uniprot\nP11111\nP22222\n", encoding="utf-8")
    out = tmp_path / "runtime_index"
    result = subprocess.run(
        [
            sys.executable,
            str(RUNTIME_BUILDER),
            "--evidence-dir",
            str(evidence),
            "--out-dir",
            str(out),
            "--target-clusters",
            str(targets),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_counts"] == {"BindingDB": 1, "ChEMBL": 1}
    licensing = manifest["evidence_licensing"]
    assert set(licensing["source_metadata"]) == {"BindingDB", "ChEMBL"}
    assert licensing["source_metadata"]["ChEMBL"]["releases"] == ["37"]
    assert licensing["source_metadata"]["ChEMBL"]["licences"] == ["CC BY-SA 3.0"]
    assert licensing["source_metadata"]["BindingDB"]["rows"] == 1
    manifest_record = licensing["source_metadata"]["ChEMBL"]["source_manifest"]
    assert Path(manifest_record["path"]) == (evidence / "source_manifest.json").resolve()
    assert manifest_record["sha256"] == _sha256(evidence / "source_manifest.json")


def test_the_runtime_builder_refuses_rows_without_source_metadata(
    tmp_path: Path,
) -> None:
    row = _runtime_row("ChEMBL", "P11111", "CCO", "1")
    row["source_license"] = ""
    evidence = _runtime_evidence(tmp_path, [row])
    targets = tmp_path / "targets.csv"
    targets.write_text("uniprot\nP11111\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(RUNTIME_BUILDER),
            "--evidence-dir",
            str(evidence),
            "--out-dir",
            str(tmp_path / "runtime_index"),
            "--target-clusters",
            str(targets),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "has no licence metadata" in result.stderr
