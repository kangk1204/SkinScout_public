"""Workbench 표적 검색 API: 단백질 → 측정 기록이 있는 화합물.

서버 판이 CLI `explore_target` 과 같은 해석·정렬을 쓰는지, 인덱스가 없을 때
조용히 빈 표가 아니라 사유를 돌려주는지를 고정한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "workbench"))

import explore_target  # noqa: E402
import server  # noqa: E402


def _write_index(base: Path, index_role: str = "production") -> Path:
    index_dir = base / "activity_retrieval_runtime_merged_test"
    index_dir.mkdir()
    (index_dir / "manifest.json").write_text(
        f'{{"index_role": "{index_role}"}}', encoding="utf-8"
    )
    edges = pd.DataFrame(
        {
            "ligand_index": [0, 1, 2, 3, 4],
            "uniprot": ["P14679", "P14679", "P14679", "P14679", "P08253"],
            "source_db": ["ChEMBL", "ChEMBL", "BindingDB", "ChEMBL", "ChEMBL"],
            "max_pactivity": [4.8, 8.1, 6.4, 5.5, 9.9],
            "median_pactivity": [4.6, 7.5, 6.0, 5.4, 9.0],
            "positive_measurement_count": [0, 5, 2, 0, 1],
            "publication_count": [5, 4, 1, 0, 1],
            "measurement_count": [2, 5, 3, 1, 1],
        }
    )
    ligands = pd.DataFrame(
        {
            "ligand_index": [0, 1, 2, 3, 4],
            "ligand_key": ["k0", "k1", "k2", "k3", "k4"],
            "standard_inchikey": [
                "AAAA-0000-0000", "BBBB-0000-0000", "CCCC-0000-0000",
                "DDDD-0000-0000", "EEEE-0000-0000",
            ],
            "canonical_smiles": ["CCO", "Cc1ccccc1", "N1CCCCC1", "CCO", "CCO"],
        }
    )
    edges.to_parquet(index_dir / "edges.parquet")
    ligands.to_parquet(index_dir / "ligands.parquet")
    return index_dir


@pytest.fixture(autouse=True)
def _clean_index_cache():
    server._TARGET_INDEX_CACHE.clear()
    yield
    server._TARGET_INDEX_CACHE.clear()


@pytest.fixture
def index_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    built = _write_index(tmp_path)
    monkeypatch.setattr(explore_target, "_discover_index_dir", lambda: built)
    return built


def _keys(result: dict) -> list[str]:
    return [row["inchikey"] for row in result["rows"]]


def test_potency_and_balanced_rank_by_measured_potency(index_dir: Path) -> None:
    balanced = server.target_binders({"target": "tyrosinase", "mode": "balanced"})
    potency = server.target_binders({"target": "tyrosinase", "mode": "potency"})

    assert balanced["resolved"] is True
    assert balanced["target"]["uniprot"] == "P14679"
    # P08253 의 9.9 는 다른 표적이라 나오면 안 된다.
    assert _keys(balanced) == _keys(potency) == [
        "BBBB-0000-0000", "CCCC-0000-0000", "DDDD-0000-0000", "AAAA-0000-0000",
    ]


def test_evidence_mode_orders_by_publication_count(index_dir: Path) -> None:
    result = server.target_binders({"target": "TYR", "mode": "evidence"})
    assert _keys(result) == [
        "AAAA-0000-0000", "BBBB-0000-0000", "CCCC-0000-0000", "DDDD-0000-0000",
    ]


def test_threshold_labels_split_at_five_and_six(index_dir: Path) -> None:
    result = server.target_binders({"target": "P14679", "mode": "potency"})
    labels = {row["inchikey"]: row["threshold"] for row in result["rows"]}
    assert labels["BBBB-0000-0000"] == "positive"      # 8.1
    assert labels["CCCC-0000-0000"] == "positive"      # 6.4
    assert labels["DDDD-0000-0000"] == "borderline"    # 5.5
    assert labels["AAAA-0000-0000"] == "below"         # 4.8


def test_unknown_name_returns_candidates_instead_of_an_empty_table(index_dir: Path) -> None:
    result = server.target_binders({"target": "p14"})
    assert result["resolved"] is False
    assert any(item["uniprot"] == "P14679" for item in result["candidates"])


def test_empty_query_asks_for_input(index_dir: Path) -> None:
    result = server.target_binders({})
    assert result["resolved"] is False
    assert "입력" in result["error"]


def test_missing_index_is_reported_not_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(explore_target, "_discover_index_dir", lambda: None)
    result = server.target_binders({"target": "tyrosinase"})
    assert result["available"] is False
    assert result["resolved"] is True
    assert "인덱스" in result["reason"]


def test_evaluation_index_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    built = _write_index(tmp_path, index_role="evaluation")
    monkeypatch.setattr(explore_target, "_discover_index_dir", lambda: built)
    result = server.target_binders({"target": "tyrosinase"})
    assert result["available"] is False
    assert "production" in result["reason"]


def test_the_index_is_loaded_once_and_reused(
    index_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}
    original = explore_target.load_index

    def counted(path: Path):
        calls["n"] += 1
        return original(path)

    monkeypatch.setattr(explore_target, "load_index", counted)
    server.target_binders({"target": "tyrosinase"})
    server.target_binders({"target": "P14679"})
    assert calls["n"] == 1


def test_limit_is_clamped(index_dir: Path) -> None:
    assert len(server.target_binders({"target": "P14679", "limit": 1})["rows"]) == 1
    huge = server.target_binders({"target": "P14679", "limit": 100})
    assert huge["rows"] and len(huge["rows"]) <= 50


def test_readiness_capability_reads_the_same_index(index_dir: Path) -> None:
    capability = server._target_search_capability()
    assert capability["ready"] is True
    assert capability["index_dir"] == str(index_dir)
