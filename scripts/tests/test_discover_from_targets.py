"""표적 리스트 발굴: 해석·커버리지·후보·등재 원료 대조·리포트 계약.

사용자가 준 표적 리스트가 표적으로 해석되지 않으면 조용히 빈 표가 나온다 -
그래서 "해석 실패"와 "인덱스 밖"을 결과에 남기는지를 고정한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import discover_from_targets as dt  # noqa: E402


def _write_index(base: Path) -> Path:
    index_dir = base / "activity_retrieval_runtime_merged_test"
    index_dir.mkdir()
    (index_dir / "manifest.json").write_text('{"index_role": "production"}', encoding="utf-8")
    pd.DataFrame(
        {
            "ligand_index": [0, 1, 2],
            "uniprot": ["P14679", "P14679", "P08253"],
            "source_db": ["ChEMBL", "ChEMBL", "BindingDB"],
            "max_pactivity": [8.1, 4.8, 6.4],
            "median_pactivity": [7.5, 4.6, 6.0],
            "positive_measurement_count": [5, 0, 2],
            "publication_count": [4, 5, 1],
            "measurement_count": [5, 2, 3],
        }
    ).to_parquet(index_dir / "edges.parquet")
    pd.DataFrame(
        {
            "ligand_index": [0, 1, 2],
            "ligand_key": ["k0", "k1", "k2"],
            "standard_inchikey": ["AJKHROAWRZBWEE-UHFFFAOYSA-N", "FOGMOMOIVJOMPM-UHFFFAOYSA-N", "ZZZZZZZZZZZZZZ-UHFFFAOYSA-N"],
            "canonical_smiles": ["CCO", "Cc1ccccc1", "N1CCCCC1"],
        }
    ).to_parquet(index_dir / "ligands.parquet")
    return index_dir


def test_accession_like_values_pass_through() -> None:
    rows = dt.resolve_targets([dt.TargetRow(label="P14679"), dt.TargetRow(label="P14679-2")])
    assert [row.uniprot for row in rows] == ["P14679", "P14679-2"]
    assert all(row.resolved for row in rows)


def test_gene_symbols_resolve_from_the_repository_lexicon() -> None:
    rows = dt.resolve_targets([dt.TargetRow(label="TYR"), dt.TargetRow(label="tyrosinase")])
    assert [row.uniprot for row in rows] == ["P14679", "P14679"]
    assert all(row.resolved for row in rows)


def test_gene_map_resolves_what_the_lexicon_does_not() -> None:
    rows = dt.resolve_targets([dt.TargetRow(label="F2RL1")], {"F2RL1": "P55085"})
    assert rows[0].uniprot == "P55085"
    assert rows[0].reason == "gene-map"


def test_unresolved_rows_keep_the_reason() -> None:
    rows = dt.resolve_targets([dt.TargetRow(label="ZZZQQ99")])
    assert rows[0].resolved is False
    assert "해석 실패" in rows[0].reason


def test_multiple_accessions_in_one_cell_expand_to_separate_targets() -> None:
    rows = dt.resolve_targets([dt.TargetRow(label="P21583; P10721", category="탈모")])
    assert [row.uniprot for row in rows] == ["P21583", "P10721"]
    assert all(row.category == "탈모" and row.resolved for row in rows)


def test_gene_map_value_with_multiple_accessions_expands() -> None:
    rows = dt.resolve_targets(
        [dt.TargetRow(label="SCF / c-KIT", category="탈모")],
        {"SCF / C-KIT": "P21583;P10721"},
    )
    assert [row.uniprot for row in rows] == ["P21583", "P10721"]
    assert rows[1].reason == "gene-map"


def test_csv_loading_finds_columns_and_keeps_categories(tmp_path: Path) -> None:
    source = tmp_path / "targets.csv"
    source.write_text("category,target,direction\n미백,TYR,억제\n광노화,MMP2,억제\n", encoding="utf-8")
    rows = dt.load_target_list(source)
    assert [row.label for row in rows] == ["TYR", "MMP2"]
    assert [row.category for row in rows] == ["미백", "광노화"]


def test_xlsx_with_an_empty_leading_row_is_read_by_header_detection(tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    source = tmp_path / "targets.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append([None, None])
    sheet.append([None, "category", "target"])
    sheet.append([None, "미백", "TYR"])
    sheet.append([None, "광노화", "MMP2"])
    workbook.save(source)

    rows = dt.load_target_list(source, target_column="target", category_column="category")
    assert [row.label for row in rows] == ["TYR", "MMP2"]
    assert [row.category for row in rows] == ["미백", "광노화"]


def test_discovery_reports_candidates_and_coverage(tmp_path: Path) -> None:
    index_dir = _write_index(tmp_path)
    rows = dt.resolve_targets(
        [
            dt.TargetRow(label="TYR", category="미백"),
            dt.TargetRow(label="P99999", category="없는 표적"),
        ]
    )
    candidates, coverage = dt.discover(index_dir, rows, top=5, mode="balanced")

    assert [c["gene"] for c in candidates] == ["TYR", "TYR"]
    assert candidates[0]["threshold"] == "positive"
    assert candidates[1]["threshold"] == "below"
    assert coverage[0]["in_index"] is True
    assert coverage[1]["in_index"] is False
    assert "인덱스 4,873 표적 밖" in coverage[1]["reason"]


def test_cosing_annotation_matches_exact_then_connectivity() -> None:
    frame = pd.DataFrame(
        {
            "inchikey": ["AJKHROAWRZBWEE-UHFFFAOYSA-N", "ZZZZZZZZZZZZZZ-UHFFFAOYSA-N"],
            "skeleton": ["AJKHROAWRZBWEE", "ZZZZZZZZZZZZZZ"],
            "inci_name": ["EXACT ONE", "CONNECTIVITY ONE"],
        }
    )
    candidates = [
        {"uniprot": "P14679", "inchikey": "AJKHROAWRZBWEE-UHFFFAOYSA-N", "cosing_match": "none", "inci_name": ""},
        {"uniprot": "P14679", "inchikey": "ZZZZZZZZZZZZZZ-UHFFFAOYSA-M", "cosing_match": "none", "inci_name": ""},
        {"uniprot": "P08253", "inchikey": "QQQQQQQQQQQQQQ-UHFFFAOYSA-N", "cosing_match": "none", "inci_name": ""},
    ]
    registered = dt.annotate_cosing(candidates, frame)

    assert [c["cosing_match"] for c in candidates] == ["exact", "connectivity", "none"]
    assert candidates[0]["inci_name"] == "EXACT ONE"
    assert registered == 1


def test_reports_are_written_and_unresolved_is_separated(tmp_path: Path) -> None:
    index_dir = _write_index(tmp_path)
    rows = dt.resolve_targets(
        [
            dt.TargetRow(label="TYR", category="미백"),
            dt.TargetRow(label="ZZZQQ99", category="미해석"),
        ]
    )
    candidates, coverage = dt.discover(index_dir, rows, top=3, mode="potency")
    out = tmp_path / "out"
    smiles_out = out / "candidates.txt"
    dt.write_reports(out, rows, candidates, coverage, 3, "potency", index_dir, smiles_out)

    assert (out / "discovery_candidates.csv").is_file()
    assert (out / "target_coverage.csv").is_file()
    unresolved = (out / "unresolved.csv").read_text(encoding="utf-8")
    assert "ZZZQQ99" in unresolved and "TYR" not in unresolved
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "입력 표적: **2**" in summary
    assert "조회 가능: **1**" in summary
    assert "실험 전 가설" in summary
    assert smiles_out.read_text(encoding="utf-8").strip()
