"""Tests for append-only performance-v2 Stage 3 attachment."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "attach_performance_v2_predictions.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("attach_performance_v2_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


attachment = _load_module()


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ranked = tmp_path / "ranked.csv"
    legacy_columns = ["target_id", "final_score", "sources"]
    legacy_rows = [
        {"target_id": "T2", "final_score": "0.9000", "sources": "a;b"},
        {"target_id": "T1", "final_score": "0.80", "sources": "c"},
    ]
    _write_csv(ranked, legacy_columns, legacy_rows)
    ranking = tmp_path / "ranking.csv"
    ranking_rows = [
        {
            "ligand_id": "query",
            "target_id": "T1",
            "performance_v2_ranking_score": "9.2",
            "performance_v2_score_semantics": "ranking_score_not_probability",
            "performance_v2_ood_route": "none",
            "performance_v2_abstained": "false",
            "performance_v2_direct_binding_probability": "0.81",
            "performance_v2_functional_modulation_probability": "0.42",
            "performance_v2_rank": "1",
        },
        {
            "ligand_id": "query",
            "target_id": "T2",
            "performance_v2_ranking_score": "8.1",
            "performance_v2_score_semantics": "ranking_score_not_probability",
            "performance_v2_ood_route": "ligand_ood",
            "performance_v2_abstained": "true",
            "performance_v2_direct_binding_probability": "",
            "performance_v2_functional_modulation_probability": "",
            "performance_v2_rank": "2",
        },
        {
            "ligand_id": "query",
            "target_id": "T3",
            "performance_v2_ranking_score": "7.0",
            "performance_v2_score_semantics": "ranking_score_not_probability",
            "performance_v2_ood_route": "none",
            "performance_v2_abstained": "false",
            "performance_v2_direct_binding_probability": "0.2",
            "performance_v2_functional_modulation_probability": "0.3",
            "performance_v2_rank": "3",
        },
    ]
    _write_csv(ranking, list(attachment.RANKING_COLUMNS), ranking_rows)
    manifest = tmp_path / "ranking.manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    model_hash = _digest("model-manifest")
    monkeypatch.setattr(
        attachment,
        "validate_ranking_manifest",
        lambda _path: {
            "outputs": {
                "ranking_csv": {"sha256": attachment.sha256_file(ranking), "rows": 3}
            },
            "inputs": {"model_manifest": {"sha256": model_hash}},
        },
    )
    return ranked, ranking, manifest, legacy_columns, legacy_rows, model_hash


def test_attachment_preserves_legacy_order_and_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ranked, ranking, manifest, legacy_columns, legacy_rows, model_hash = _fixture(
        tmp_path, monkeypatch
    )
    out = tmp_path / "attached.csv"
    out_manifest = tmp_path / "attached.manifest.json"
    attachment.attach(
        ranked_csv=ranked,
        ranking_csv=ranking,
        ranking_manifest_path=manifest,
        out_csv=out,
        out_manifest=out_manifest,
        candidate_id="P1",
    )

    with out.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert reader.fieldnames == [*legacy_columns, *attachment.APPENDED_COLUMNS]
    assert [{key: row[key] for key in legacy_columns} for row in rows] == legacy_rows
    assert rows[0]["target_id"] == "T2"
    assert rows[0]["performance_v2_model_rank"] == "2"
    assert rows[0]["performance_v2_coverage_status"] == "abstained_ood"
    assert rows[1]["performance_v2_model_rank"] == "1"
    assert rows[1]["performance_v2_direct_binding_probability"] == "0.81"
    assert all(row["performance_v2_promotion_status"] == "research_candidate_not_promoted" for row in rows)
    payload = json.loads(out_manifest.read_text(encoding="utf-8"))
    assert payload["append_only"] is True
    assert payload["legacy_order_preserved"] is True
    assert payload["counts"] == {
        "abstained_targets": 1,
        "attached_targets": 2,
        "legacy_targets": 2,
        "v2_universe_targets": 3,
    }
    assert payload["inputs"]["performance_v2_model_manifest_sha256"] == model_hash


def test_missing_target_fails_without_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ranked, ranking, manifest, *_ = _fixture(tmp_path, monkeypatch)
    rows = list(csv.DictReader(ranking.open(newline="", encoding="utf-8")))
    _write_csv(ranking, list(attachment.RANKING_COLUMNS), rows[1:])
    monkeypatch.setattr(
        attachment,
        "validate_ranking_manifest",
        lambda _path: {
            "outputs": {"ranking_csv": {"sha256": attachment.sha256_file(ranking)}},
            "inputs": {"model_manifest": {"sha256": _digest("model")}},
        },
    )
    out = tmp_path / "attached.csv"
    out_manifest = tmp_path / "attached.json"
    with pytest.raises(attachment.AttachmentError, match="missing legacy targets"):
        attachment.attach(
            ranked_csv=ranked,
            ranking_csv=ranking,
            ranking_manifest_path=manifest,
            out_csv=out,
            out_manifest=out_manifest,
            candidate_id="P1",
        )
    assert not out.exists()
    assert not out_manifest.exists()


def test_existing_v2_column_and_duplicate_target_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ranked, ranking, manifest, *_ = _fixture(tmp_path, monkeypatch)
    rows = list(csv.DictReader(ranked.open(newline="", encoding="utf-8")))
    columns = ["target_id", "final_score", "sources", attachment.APPENDED_COLUMNS[0]]
    for row in rows:
        row[attachment.APPENDED_COLUMNS[0]] = "stale"
    _write_csv(ranked, columns, rows)
    with pytest.raises(attachment.AttachmentError, match="already contain"):
        attachment.attach(
            ranked_csv=ranked,
            ranking_csv=ranking,
            ranking_manifest_path=manifest,
            out_csv=tmp_path / "out.csv",
            out_manifest=tmp_path / "out.json",
            candidate_id="P1",
        )


def test_cli_preserves_stale_outputs_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ranked, ranking, manifest, *_ = _fixture(tmp_path, monkeypatch)
    out = tmp_path / "out.csv"
    out_manifest = tmp_path / "out.json"
    out.write_text("stale\n", encoding="utf-8")
    out_manifest.write_text("stale\n", encoding="utf-8")
    ranked.write_text("broken\n", encoding="utf-8")
    with pytest.raises(attachment.AttachmentError, match="refusing to overwrite"):
        attachment.main(
            [
                "--ranked-csv",
                str(ranked),
                "--ranking-csv",
                str(ranking),
                "--ranking-manifest",
                str(manifest),
                "--out-csv",
                str(out),
                "--out-manifest",
                str(out_manifest),
            ]
        )
    assert out.read_text(encoding="utf-8") == "stale\n"
    assert out_manifest.read_text(encoding="utf-8") == "stale\n"
