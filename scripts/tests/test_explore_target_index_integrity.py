"""표적 우선 검색 loader: manifest의 SHA-256·행 수까지 대조한다.

role/schema/열만 보면 다른 실행의 같은 schema parquet으로 바꿔도 검색이 계속된다.
compound-first reference loader와 같은 hash·행 수 계약을 소비 경로에도 건다.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import discover_from_targets as dt  # noqa: E402
import explore_target as et  # noqa: E402
from activity_retrieval_scoring import PRODUCTION_INDEX_SCHEMA  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_index(base: Path, *, extra: bool = False) -> Path:
    index_dir = base / "activity_retrieval_index"
    index_dir.mkdir(parents=True)
    edges = pd.DataFrame(
        {
            "ligand_index": [0, 1],
            "uniprot": ["P14679", "P08253"],
            "source_db": ["ChEMBL", "BindingDB"],
            "max_pactivity": [8.0, 6.0],
            "median_pactivity": [7.0, 5.5],
            "positive_measurement_count": [3, 1],
            "publication_count": [2, 1],
            "measurement_count": [3, 2],
        }
    )
    ligands = pd.DataFrame(
        {
            "ligand_index": [0, 1],
            "ligand_key": ["k0", "k1"],
            "standard_inchikey": ["AAAA-0000-0000", "BBBB-0000-0000"],
            "canonical_smiles": ["CCO", "N1CCCCC1"],
        }
    )
    if extra:
        edges.loc[len(edges)] = [2, "P99999", "ChEMBL", 7.0, 6.0, 2, 1, 2]
        ligands.loc[len(ligands)] = [2, "k2", "CCCC-0000-0000", "CCC"]
    edges.to_parquet(index_dir / "edges.parquet")
    ligands.to_parquet(index_dir / "ligands.parquet")
    manifest = {
        "index_role": "production",
        "schema_version": PRODUCTION_INDEX_SCHEMA,
        "outputs": {
            "edges": {
                "sha256": _sha256(index_dir / "edges.parquet"),
                "rows": len(edges),
            },
            "ligands": {
                "sha256": _sha256(index_dir / "ligands.parquet"),
                "rows": len(ligands),
            },
        },
    }
    (index_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return index_dir


def _read_manifest(index_dir: Path) -> dict:
    return json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))


def _write_manifest(index_dir: Path, manifest: dict) -> None:
    (index_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_registered_index_loads(tmp_path: Path) -> None:
    index_dir = _write_index(tmp_path)
    edges, ligands = et.load_index(index_dir)
    assert len(edges) == 2
    assert len(ligands) == 2


def test_swapped_same_schema_parquet_is_rejected(tmp_path: Path) -> None:
    index_dir = _write_index(tmp_path)
    other = _write_index(tmp_path / "other", extra=True)
    (index_dir / "edges.parquet").write_bytes((other / "edges.parquet").read_bytes())

    with pytest.raises(SystemExit, match="sha256 mismatch"):
        et.load_index(index_dir)


def test_deleted_row_is_rejected(tmp_path: Path) -> None:
    index_dir = _write_index(tmp_path)
    edges = pd.read_parquet(index_dir / "edges.parquet").iloc[:-1]
    edges.to_parquet(index_dir / "edges.parquet")

    with pytest.raises(SystemExit, match="sha256 mismatch"):
        et.load_index(index_dir)


def test_changed_bytes_are_rejected(tmp_path: Path) -> None:
    index_dir = _write_index(tmp_path)
    payload = bytearray((index_dir / "ligands.parquet").read_bytes())
    payload[len(payload) // 2] ^= 0xFF
    (index_dir / "ligands.parquet").write_bytes(bytes(payload))

    with pytest.raises(SystemExit, match="sha256 mismatch"):
        et.load_index(index_dir)


def test_row_count_mismatch_is_rejected_even_with_recorded_hash(
    tmp_path: Path,
) -> None:
    index_dir = _write_index(tmp_path)
    manifest = _read_manifest(index_dir)
    manifest["outputs"]["edges"]["rows"] = 99
    _write_manifest(index_dir, manifest)

    with pytest.raises(SystemExit, match="rows mismatch"):
        et.load_index(index_dir)


def test_missing_outputs_record_fails_closed(tmp_path: Path) -> None:
    index_dir = _write_index(tmp_path)
    manifest = _read_manifest(index_dir)
    del manifest["outputs"]
    _write_manifest(index_dir, manifest)

    with pytest.raises(SystemExit, match="outputs.edges"):
        et.load_index(index_dir)


def test_target_first_consumer_path_rejects_tampered_index(tmp_path: Path) -> None:
    """소비자 경로(discover_from_targets.discover)도 같은 검증을 지나야 한다."""
    index_dir = _write_index(tmp_path)
    manifest = _read_manifest(index_dir)
    manifest["outputs"]["ligands"]["sha256"] = "0" * 64
    _write_manifest(index_dir, manifest)

    with pytest.raises(SystemExit, match="sha256 mismatch"):
        dt.discover(index_dir, [dt.TargetRow(label="P14679")], top=5, mode="balanced")
