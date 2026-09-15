"""explore_target: 단백질 이름으로 시작해 알려진 결합 화합물을 찾는 도구의 계약."""

from __future__ import annotations

import subprocess
import sys

import pandas as pd
import pytest

ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import explore_target as et  # noqa: E402

SCRIPT = ROOT / "scripts" / "explore_target.py"


def production_index() -> bool:
    return et._discover_index_dir() is not None


# ----------------------------------------------------------------------------- 이름 해석


def test_exact_uniprot_resolves_to_curated_entry() -> None:
    entry, rest = et.resolve_target("P14679")
    assert rest == []
    assert entry is not None
    assert entry.uniprot == "P14679"
    assert entry.gene == "TYR"


def test_gene_symbol_is_case_insensitive() -> None:
    assert et.resolve_target("tyr")[0].uniprot == "P14679"
    assert et.resolve_target("MMP2")[0].uniprot == "P08253"


def test_common_name_alias_resolves() -> None:
    assert et.resolve_target("tyrosinase")[0].uniprot == "P14679"
    assert et.resolve_target("retinoic acid receptor alpha")[0].uniprot == "P10276"
    assert et.resolve_target("retinoic acid receptor alpha")[0].gene == "RARA"


def test_partial_query_returns_candidates() -> None:
    entry, candidates = et.resolve_target("p14")
    assert entry is None
    assert "P14679" in candidates


def test_garbage_query_has_no_candidates() -> None:
    entry, candidates = et.resolve_target("zzzznope____")
    assert entry is None
    assert candidates == []


def test_empty_query_has_no_candidates() -> None:
    entry, candidates = et.resolve_target("  ")
    assert entry is None
    assert candidates == []


# ----------------------------------------------------------------------------- 순위


def _mini_index() -> tuple[pd.DataFrame, pd.DataFrame]:
    edges = pd.DataFrame(
        {
            "ligand_index": [0, 1, 2, 3],
            "uniprot": ["P14679", "P14679", "P14679", "P08253"],
            "source_db": ["ChEMBL", "ChEMBL", "BindingDB", "ChEMBL"],
            "max_pactivity": [6.5, 8.0, 7.2, 10.0],
            "median_pactivity": [6.0, 7.5, 6.9, 9.0],
            "positive_measurement_count": [1, 3, 1, 5],
            "publication_count": [1, 2, 1, 4],
            "measurement_count": [1, 3, 1, 5],
        }
    )
    ligands = pd.DataFrame(
        {
            "ligand_index": [0, 1, 2, 3],
            "ligand_key": ["k0", "k1", "k2", "k3"],
            "canonical_smiles": ["C1CCCCC1", "Cc1ccccc1", "N1CCCCC1", "CCO"],
        }
    )
    return edges, ligands


def test_rank_orders_by_potency_and_respects_top() -> None:
    edges, ligands = _mini_index()
    frame = et.rank_target(edges, ligands, "P14679", top=2, mode="potency")
    assert list(frame["ligand_key"]) == ["k1", "k2"]


def test_evidence_mode_orders_by_publication_count() -> None:
    edges, ligands = _mini_index()
    frame = et.rank_target(edges, ligands, "P14679", top=3, mode="evidence")
    assert list(frame["ligand_key"]) == ["k1", "k2", "k0"]


def test_unknown_target_returns_empty() -> None:
    edges, ligands = _mini_index()
    frame = et.rank_target(edges, ligands, "P99999", top=5, mode="balanced")
    assert frame.empty


def test_empty_target_table_mentions_coverage() -> None:
    edges, ligands = _mini_index()
    frame = et.rank_target(edges, ligands, "P99999", top=5, mode="balanced")
    lines = et.format_target_table(frame, "P99999", None)
    assert any("측정 활성이 없습니다" in line for line in lines)


def test_pipeline_commands_use_fast_default() -> None:
    edges, ligands = _mini_index()
    frame = et.rank_target(edges, ligands, "P08253", top=2, mode="balanced")
    cmds = et.pipeline_commands(frame, "fast", limit=1)
    assert len(cmds) == 1
    assert cmds[0] == "python scripts/run_skinscout.py --smiles 'CCO' --mode fast"


# ----------------------------------------------------------------------------- 인덱스 계약


def test_load_index_rejects_evaluation_role(tmp_path) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "manifest.json").write_text('{"index_role": "evaluation"}', encoding="utf-8")
    pd.DataFrame().to_parquet(index_dir / "edges.parquet")
    pd.DataFrame().to_parquet(index_dir / "ligands.parquet")
    with pytest.raises(SystemExit, match="production"):
        et.load_index(index_dir)


def test_load_index_complains_about_missing_files(tmp_path) -> None:
    index_dir = tmp_path / "empty"
    index_dir.mkdir()
    with pytest.raises(SystemExit, match="불완전"):
        et.load_index(index_dir)


def test_discovery_prefers_the_pipeline_config_index() -> None:
    """파이프라인이 실제로 읽는 인덱스와 다른 판을 조회하면 안 된다."""
    configured = et._configured_index_dir()
    if configured is None:
        pytest.skip("workflow/config.yaml 또는 로컬 검색 인덱스가 없습니다")
    assert et._discover_index_dir() == configured


@pytest.mark.skipif(not production_index(), reason="로컬 검색 인덱스가 없습니다")
def test_end_to_end_uniprot_lookup() -> None:
    res = subprocess.run(
        [sys.executable, str(SCRIPT), "P14679", "--top", "3"],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr
    assert "표적 P14679" in res.stdout
    assert "run_skinscout.py --smiles" in res.stdout


@pytest.mark.skipif(not production_index(), reason="로컬 검색 인덱스가 없습니다")
def test_smiles_out_writes_pipeline_input(tmp_path) -> None:
    out = tmp_path / "ligands.txt"
    res = subprocess.run(
        [sys.executable, str(SCRIPT), "P10276", "--smiles-out", str(out), "--top", "2"],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert all(line.strip() and " " not in line for line in lines)


def test_unknown_name_is_a_helpful_error() -> None:
    res = subprocess.run(
        [sys.executable, str(SCRIPT), "zzzznope____"], capture_output=True, text=True,
    )
    assert res.returncode != 0
    assert "해석할 수 없습니다" in res.stderr