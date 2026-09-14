from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from scripts.build_target_sequence_embeddings import (  # noqa: E402
    FairEsmEmbedder,
    build_target_sequence_embeddings,
)


class FakeEmbedder:
    model_name = "fake-esm"
    dimension = 3
    representation_layer = 6
    checkpoint_sha256 = "a" * 64

    def __init__(
        self, *, bad: dict[str, np.ndarray] | None = None, fail: bool = False
    ) -> None:
        self.bad = bad or {}
        self.fail = fail
        self.calls: list[list[tuple[str, str]]] = []

    def embed_chunks(self, chunks: list[tuple[str, str]]) -> dict[str, np.ndarray]:
        if self.fail:
            raise RuntimeError("fake embed failure")
        self.calls.append(chunks)
        output: dict[str, np.ndarray] = {}
        for chunk_id, sequence in chunks:
            output[chunk_id] = self.bad.get(
                chunk_id,
                np.array([len(sequence), sequence.count("A") + 1, 1.0], dtype=np.float32),
            )
        return output


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, ids: list[str]) -> None:
    path.write_text(
        "uniprot,target_cluster_30,target_cluster_50\n"
        + "".join(f"{x},{x},{x}\n" for x in ids)
    )


def _fixtures(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    target_csv = tmp_path / "screenable.csv"
    base_fasta = tmp_path / "base.fasta"
    supplemental_fasta = tmp_path / "supplemental.fasta"
    manifest = tmp_path / "screenable.manifest.json"
    _write_csv(target_csv, ["P2", "P1", "P3"])
    base_fasta.write_text(">sp|P2|Protein two\nCCCC\n>P1\nAAAAAA\n")
    supplemental_fasta.write_text(">P3 supplemental\nGGGGG\n")
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "skinscout.screenable-target-cluster-map.v2",
                "universe_policy": {
                    "evaluation_panel_used": False,
                    "known_target_assistance": False,
                },
                "production_contract": {"passes": True},
                "artifact": {"path": str(target_csv), "sha256": _sha256(target_csv), "rows": 3},
                "inputs": {
                    "base_target_csv": {
                        "path": str(target_csv),
                        "sha256": _sha256(target_csv),
                        "fasta_path": str(base_fasta),
                        "fasta_sha256": _sha256(base_fasta),
                        "source_manifest": {
                            "sequence_role": "canonical_uniprot_reference_sequence"
                        },
                    },
                    "supplemental_fasta": {
                        "path": str(supplemental_fasta),
                        "sha256": _sha256(supplemental_fasta),
                    },
                },
            }
        )
        + "\n"
    )
    return (
        target_csv,
        manifest,
        tmp_path / "embeddings.parquet",
        tmp_path / "embeddings.manifest.json",
    )


def _build(tmp_path: Path, embedder: FakeEmbedder | None = None) -> dict[str, object]:
    target_csv, manifest, out_parquet, out_manifest = _fixtures(tmp_path)
    return build_target_sequence_embeddings(
        target_csv=target_csv,
        target_manifest=manifest,
        out_parquet=out_parquet,
        out_manifest=out_manifest,
        embedder=embedder or FakeEmbedder(),
        max_chunk_residues=4,
        tokens_per_batch=8,
    )


def test_builds_sorted_fixed_size_embeddings_and_manifest(tmp_path: Path) -> None:
    embedder = FakeEmbedder()

    manifest = _build(tmp_path, embedder)

    table = pq.read_table(tmp_path / "embeddings.parquet")
    assert table.column_names == ["uniprot", "embedding"]
    assert pa.types.is_fixed_size_list(table.schema.field("embedding").type)
    assert table.schema.field("embedding").type.list_size == 3
    assert table.column("uniprot").to_pylist() == ["P1", "P2", "P3"]
    embeddings = table.column("embedding").to_pylist()
    assert embeddings[0] == pytest.approx([20.0 / 6.0, 26.0 / 6.0, 1.0])
    assert embeddings[1] == pytest.approx([4.0, 1.0, 1.0])
    assert embeddings[2] == pytest.approx([3.4, 1.0, 1.0])
    payload = json.loads((tmp_path / "embeddings.manifest.json").read_text())
    assert payload == manifest
    assert payload["schema_version"] == "skinscout.target-sequence-embeddings.v1"
    assert payload["source_schema_version"] == "skinscout.screenable-target-cluster-map.v2"
    assert payload["universe_policy"] == {
        "evaluation_panel_used": False,
        "known_target_assistance": False,
        "production_contract_passes": True,
    }
    assert payload["contract"] == {
        "evaluation_panel_used": False,
        "known_target_assistance": False,
        "training_labels_used": False,
    }
    assert payload["inputs"]["target_clusters"] == {
        "path": str((tmp_path / "screenable.csv").resolve()),
        "sha256": _sha256(tmp_path / "screenable.csv"),
        "rows": 3,
    }
    assert payload["inputs"]["target_cluster_manifest"] == {
        "path": str((tmp_path / "screenable.manifest.json").resolve()),
        "sha256": _sha256(tmp_path / "screenable.manifest.json"),
    }
    assert payload["model"]["name"] == "fake-esm"
    assert payload["model"]["representation_layer"] == 6
    assert payload["model"]["checkpoint_sha256"] == "a" * 64
    assert payload["embedding"]["chunk_policy"]["max_chunk_residues"] == 4
    assert payload["artifact"]["sha256"] == _sha256(tmp_path / "embeddings.parquet")
    assert payload["artifact"]["rows"] == 3
    assert payload["outputs"]["embeddings"] == {
        "path": str((tmp_path / "embeddings.parquet").resolve()),
        "sha256": _sha256(tmp_path / "embeddings.parquet"),
        "rows": 3,
        "dimension": 3,
    }
    assert sorted(chunk_id for call in embedder.calls for chunk_id, _ in call) == [
        "P1:0",
        "P1:1",
        "P2:0",
        "P3:0",
        "P3:1",
    ]
    for call in embedder.calls:
        padded_tokens = len(call) * max(len(sequence) + 2 for _, sequence in call)
        assert padded_tokens <= 8


def test_checkpoint_hash_uses_exact_model_file(tmp_path: Path) -> None:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    model_name = "esm2_t6_8M_UR50D"
    checkpoint = checkpoints / f"{model_name}.pt"
    checkpoint.write_bytes(b"main model")
    (checkpoints / f"{model_name}-contact-regression.pt").write_bytes(b"auxiliary weights")
    embedder = FairEsmEmbedder.__new__(FairEsmEmbedder)
    embedder.model_name = model_name
    embedder._torch = SimpleNamespace(hub=SimpleNamespace(get_dir=lambda: str(tmp_path)))

    assert embedder._checkpoint_sha256() == _sha256(checkpoint)


@pytest.mark.parametrize(
    ("target_ids", "base_text", "supplemental_text", "message"),
    [
        (
            ["P1", "P2", "P3"],
            ">P1\nAAAA\n>P2\nCCCC\n",
            ">P4\nGGGG\n",
            "missing=['P3'] extra=['P4']",
        ),
        (["P1", "P2"], ">P1\nAAAA\n>P1\nCCCC\n", ">P2\nGGGG\n", "Duplicate FASTA accession"),
        (["P1", "P2"], ">P1\nAAAA\n", ">P2\nGGGG\n>P3\nTTTT\n", "missing=[] extra=['P3']"),
    ],
)
def test_rejects_missing_duplicate_and_extra_fasta_targets(
    tmp_path: Path,
    target_ids: list[str],
    base_text: str,
    supplemental_text: str,
    message: str,
) -> None:
    target_csv, manifest, out_parquet, out_manifest = _fixtures(tmp_path)
    _write_csv(target_csv, target_ids)
    base = tmp_path / "base.fasta"
    supplemental = tmp_path / "supplemental.fasta"
    base.write_text(base_text)
    supplemental.write_text(supplemental_text)
    payload = json.loads(manifest.read_text())
    payload["artifact"]["sha256"] = _sha256(target_csv)
    payload["artifact"]["rows"] = len(target_ids)
    payload["inputs"]["base_target_csv"]["fasta_sha256"] = _sha256(base)
    payload["inputs"]["supplemental_fasta"]["sha256"] = _sha256(supplemental)
    manifest.write_text(json.dumps(payload) + "\n")

    with pytest.raises(SystemExit, match=re_escape(message)):
        build_target_sequence_embeddings(
            target_csv=target_csv,
            target_manifest=manifest,
            out_parquet=out_parquet,
            out_manifest=out_manifest,
            embedder=FakeEmbedder(),
        )


def test_rejects_stale_manifest_hash(tmp_path: Path) -> None:
    target_csv, manifest, out_parquet, out_manifest = _fixtures(tmp_path)
    base = tmp_path / "base.fasta"
    base.write_text(base.read_text() + ">PX\nTTTT\n")

    with pytest.raises(SystemExit, match="base_target_csv.fasta_path sha256 is stale"):
        build_target_sequence_embeddings(
            target_csv=target_csv,
            target_manifest=manifest,
            out_parquet=out_parquet,
            out_manifest=out_manifest,
            embedder=FakeEmbedder(),
        )


@pytest.mark.parametrize(
    ("bad_vector", "message"),
    [
        (np.array([1.0, np.nan, 1.0], dtype=np.float32), "non-finite"),
        (np.array([0.0, 0.0, 0.0], dtype=np.float32), "zero"),
        (np.array([1.0, 2.0], dtype=np.float32), "dimension"),
    ],
)
def test_rejects_nonfinite_zero_and_wrong_dim_embeddings(
    tmp_path: Path, bad_vector: np.ndarray, message: str
) -> None:
    _fixtures(tmp_path)
    target_csv = tmp_path / "screenable.csv"
    manifest = tmp_path / "screenable.manifest.json"
    out_parquet = tmp_path / "embeddings.parquet"
    out_manifest = tmp_path / "embeddings.manifest.json"
    embedder = FakeEmbedder(bad={"P1:0": bad_vector})

    with pytest.raises(SystemExit, match=message):
        build_target_sequence_embeddings(
            target_csv=target_csv,
            target_manifest=manifest,
            out_parquet=out_parquet,
            out_manifest=out_manifest,
            embedder=embedder,
            max_chunk_residues=4,
        )


def test_removes_stale_outputs_on_failure(tmp_path: Path) -> None:
    target_csv, manifest, out_parquet, out_manifest = _fixtures(tmp_path)
    out_parquet.write_text("stale parquet")
    out_manifest.write_text("stale manifest")

    with pytest.raises(RuntimeError, match="fake embed failure"):
        build_target_sequence_embeddings(
            target_csv=target_csv,
            target_manifest=manifest,
            out_parquet=out_parquet,
            out_manifest=out_manifest,
            embedder=FakeEmbedder(fail=True),
        )

    assert not out_parquet.exists()
    assert not out_manifest.exists()


@pytest.mark.parametrize("protected_name", ["target_csv", "base_fasta"])
def test_rejects_output_alias_without_modifying_input(
    tmp_path: Path, protected_name: str
) -> None:
    target_csv, manifest, _out_parquet, out_manifest = _fixtures(tmp_path)
    protected = {
        "target_csv": target_csv,
        "base_fasta": tmp_path / "base.fasta",
    }[protected_name]
    before = protected.read_bytes()

    with pytest.raises(SystemExit, match="aliases a protected input"):
        build_target_sequence_embeddings(
            target_csv=target_csv,
            target_manifest=manifest,
            out_parquet=protected,
            out_manifest=out_manifest,
            embedder=FakeEmbedder(),
        )

    assert protected.read_bytes() == before
    assert not out_manifest.exists()


def re_escape(text: str) -> str:
    return re.escape(text)
