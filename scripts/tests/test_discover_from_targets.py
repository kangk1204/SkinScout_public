"""표적 리스트 발굴: 해석·커버리지·후보·등재 원료 대조·리포트 계약.

사용자가 준 표적 리스트가 표적으로 해석되지 않으면 조용히 빈 표가 나온다 -
그래서 "해석 실패"와 "인덱스 밖"을 결과에 남기는지를 고정한다.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import discover_from_targets as dt  # noqa: E402
from activity_retrieval_scoring import PRODUCTION_INDEX_SCHEMA  # noqa: E402


def _write_index(base: Path) -> Path:
    index_dir = base / "activity_retrieval_runtime_merged_test"
    index_dir.mkdir()
    edges = pd.DataFrame(
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
    )
    ligands = pd.DataFrame(
        {
            "ligand_index": [0, 1, 2],
            "ligand_key": ["k0", "k1", "k2"],
            "standard_inchikey": ["AJKHROAWRZBWEE-UHFFFAOYSA-N", "FOGMOMOIVJOMPM-UHFFFAOYSA-N", "ZZZZZZZZZZZZZZ-UHFFFAOYSA-N"],
            "canonical_smiles": ["CCO", "Cc1ccccc1", "N1CCCCC1"],
        }
    )
    edges.to_parquet(index_dir / "edges.parquet")
    ligands.to_parquet(index_dir / "ligands.parquet")
    (index_dir / "manifest.json").write_text(
        json.dumps(
            {
                "index_role": "production",
                "schema_version": PRODUCTION_INDEX_SCHEMA,
                "outputs": {
                    "edges": {
                        "sha256": hashlib.sha256(
                            (index_dir / "edges.parquet").read_bytes()
                        ).hexdigest(),
                        "rows": len(edges),
                    },
                    "ligands": {
                        "sha256": hashlib.sha256(
                            (index_dir / "ligands.parquet").read_bytes()
                        ).hexdigest(),
                        "rows": len(ligands),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
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


def test_user_gene_map_takes_precedence_over_the_repository_lexicon() -> None:
    rows = dt.resolve_targets([dt.TargetRow(label="TYR")], {"TYR": "P00001"})

    assert rows[0].resolved is True
    assert rows[0].uniprot == "P00001"
    assert rows[0].reason == "gene-map"


def test_generic_protein_names_stay_ambiguous_with_candidates() -> None:
    rows = dt.resolve_targets([dt.TargetRow(label="Collagen", category="테스트")])

    assert rows[0].resolved is False
    assert len(rows[0].candidates) > 1
    assert "Q9UMD9" in rows[0].candidates
    assert "모호" in rows[0].reason
    assert "P02461" in rows[0].reason or "Q9UMD9" in rows[0].reason


class _JsonResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False


def _fake_response(payload: dict) -> _JsonResponse:
    return _JsonResponse(json.dumps(payload).encode("utf-8"))


def test_online_exact_primary_gene_match_resolves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """gene_exact catches synonyms (FLG is one for FGFR1). Only the exact
    primary gene name may resolve, never the first result."""
    payload = {
        "results": [
            {
                "primaryAccession": "P11362",
                "genes": [
                    {"geneName": {"value": "FGFR1"}, "synonyms": [{"value": "FLG"}]}
                ],
            },
            {"primaryAccession": "P20930", "genes": [{"geneName": {"value": "FLG"}}]},
        ]
    }
    monkeypatch.setattr(
        dt.urllib.request, "urlopen", lambda *_args, **_kwargs: _fake_response(payload)
    )

    accession, state, candidates = dt._online_gene_lookup(
        "FLG", {}, now=0.0, sleep=lambda _seconds: None
    )

    assert (accession, state, candidates) == ("P20930", "resolved", [])


def test_online_synonym_only_hits_are_ambiguous_not_adopted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "results": [
            {"primaryAccession": "P11111", "genes": [{"geneName": {"value": "OTHER"}}]},
            {"primaryAccession": "P22222", "genes": [{"geneName": {"value": "ANOTHER"}}]},
        ]
    }
    monkeypatch.setattr(
        dt.urllib.request, "urlopen", lambda *_args, **_kwargs: _fake_response(payload)
    )

    accession, state, candidates = dt._online_gene_lookup(
        "SYNONYM", {}, now=0.0, sleep=lambda _seconds: None
    )

    assert accession is None
    assert state == "ambiguous"
    assert candidates == ["P11111", "P22222"]


def test_online_transient_failure_is_retried_and_cached_with_ttl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from urllib.error import URLError

    calls: list[float | None] = []

    def failing_urlopen(_request, timeout=None):
        calls.append(timeout)
        raise URLError("temporary network failure")

    monkeypatch.setattr(dt.urllib.request, "urlopen", failing_urlopen)
    sleeps: list[float] = []
    cache: dict[str, object] = {}

    assert dt._online_gene_lookup(
        "NEWGENE", cache, now=100.0, sleep=sleeps.append
    ) == (None, "transient", [])
    assert len(calls) == dt.ONLINE_LOOKUP_ATTEMPTS
    assert sleeps, "transient failures must back off between attempts"
    assert cache["NEWGENE"] == {"state": "transient", "at": 100.0}
    assert cache["NEWGENE"] is not None, "a timeout must not become a permanent null"

    calls.clear()
    assert dt._online_gene_lookup(
        "NEWGENE",
        cache,
        now=100.0 + dt.TRANSIENT_CACHE_SECONDS - 1,
        sleep=sleeps.append,
    )[1] == "transient"
    assert calls == [], "the cached transient answer is reused inside its TTL"

    assert dt._online_gene_lookup(
        "NEWGENE",
        cache,
        now=100.0 + dt.TRANSIENT_CACHE_SECONDS + 1,
        sleep=sleeps.append,
    )[1] == "transient"
    assert calls, "an expired transient cache entry must be retried"


def test_transient_online_failure_marks_the_row_for_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        dt, "_online_gene_lookup", lambda *_args, **_kwargs: (None, "transient", [])
    )

    rows = dt.resolve_targets(
        [dt.TargetRow(label="ZZNEWGENE")], resolve_online=True
    )

    assert rows[0].resolved is False
    assert rows[0].transient is True
    assert "일시 실패" in rows[0].reason


def test_a_404_is_cached_as_a_permanent_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from urllib.error import HTTPError

    calls: list[object] = []

    def missing_urlopen(_request, timeout=None):
        calls.append(timeout)
        raise HTTPError("https://example.test", 404, "Not Found", {}, None)

    monkeypatch.setattr(dt.urllib.request, "urlopen", missing_urlopen)
    cache: dict[str, object] = {}

    assert dt._online_gene_lookup("NOPE", cache, now=0.0) == (None, "not_found", [])
    assert cache["NOPE"] is None
    assert dt._online_gene_lookup("NOPE", cache, now=0.0) == (None, "not_found", [])
    assert len(calls) == 1
