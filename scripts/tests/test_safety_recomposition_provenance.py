"""C21 regression: safety recomposition must verify batch manifests and the
bytes, schema, model, and request identity of every source response."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from reanalyze_target_candidate_safety import (
    reconcile_safety_batches,
    sha256_file,
    sha256_text,
)

SCHEMA = "skinscout.target-candidate-safety.v1"
MODEL = "husspred"
COMPOUND = "structure-sha256:" + "d" * 64


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build(
    tmp_path: Path,
    *,
    envelope_model: str = MODEL,
    request_smiles: str = "CCO",
    request_hash: str | None = None,
    row_smiles: str | None = None,
    row_hash: str | None = None,
    candidate_count: int | None = None,
) -> tuple[list[Path], dict[str, Path]]:
    paths: list[Path] = []
    artifacts: dict[str, Path] = {}
    canonical = row_smiles if row_smiles is not None else request_smiles
    for name, call in (("first", "negative"), ("second", "positive")):
        directory = tmp_path / name
        artifact = directory / "raw" / f"{envelope_model}.json"
        _write_json(
            artifact,
            {
                "schema": SCHEMA + ".model-response",
                "model": envelope_model,
                "execution": "live_refresh",
                "retrieved_at": "2026-09-15T00:00:00+00:00",
                "request_smiles": request_smiles,
                "request_smiles_sha256": request_hash
                if request_hash is not None
                else sha256_text(request_smiles),
                "parsed_payload": {
                    "smiles": request_smiles,
                    "status": "ok",
                    "skin_sens_call": call,
                },
            },
        )
        evidence = {
            "source_url": "https://example.invalid",
            "execution": "live_refresh",
            "retrieved_at": "2026-09-15T00:00:00+00:00",
            "request_smiles_sha256": sha256_text(request_smiles),
            "artifact_path": str(artifact),
            "artifact_sha256": sha256_file(artifact),
        }
        row = {
            "compound_id": COMPOUND,
            "canonical_smiles": canonical,
            "canonical_smiles_sha256": row_hash
            if row_hash is not None
            else sha256_text(canonical),
            "decision": "FLAG_HIGH",
            "safety_consensus_valid": False,
            "full_analysis_complete": False,
            "run_valid": False,
            "provenance_complete": False,
            "applicability": {"verdict": "in_scope"},
            "admet_status": "unavailable",
            "model_results": {MODEL: evidence},
        }
        path = directory / "candidate_safety.jsonl"
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        _write_json(
            directory / "manifest.json",
            {
                "schema": SCHEMA,
                "candidate_count": candidate_count
                if candidate_count is not None
                else 1,
                "artifacts": {
                    "candidate_safety_jsonl": {
                        "path": str(path),
                        "sha256": sha256_file(path),
                    }
                },
            },
        )
        paths.append(path)
        artifacts[name] = artifact
    return paths, artifacts


def test_normal_recomposition_selects_validated_evidence(tmp_path: Path) -> None:
    paths, _artifacts = _build(tmp_path)
    rows, comparisons = reconcile_safety_batches(paths)
    assert len(rows) == 1
    assert len(comparisons) == 1
    assert rows[0]["model_results"][MODEL]["artifact_path"].endswith("husspred.json")


def test_artifact_byte_change_is_rejected(tmp_path: Path) -> None:
    paths, artifacts = _build(tmp_path)
    artifacts["second"].write_bytes(artifacts["second"].read_bytes() + b" ")
    with pytest.raises(SystemExit, match="hash mismatch"):
        reconcile_safety_batches(paths)


def test_model_swap_is_rejected(tmp_path: Path) -> None:
    paths, _artifacts = _build(tmp_path, envelope_model="stoptox")
    with pytest.raises(SystemExit, match="model mismatch"):
        reconcile_safety_batches(paths)


def test_request_hash_swap_is_rejected(tmp_path: Path) -> None:
    paths, _artifacts = _build(tmp_path, request_hash="0" * 64)
    with pytest.raises(SystemExit, match="request hash mismatch"):
        reconcile_safety_batches(paths)


def test_request_identity_swap_is_rejected(tmp_path: Path) -> None:
    paths, _artifacts = _build(
        tmp_path,
        request_smiles="CCN",
        request_hash=sha256_text("CCO"),
        row_smiles="CCO",
    )
    with pytest.raises(SystemExit, match="request identity mismatch"):
        reconcile_safety_batches(paths)


def test_jsonl_byte_change_is_rejected(tmp_path: Path) -> None:
    paths, _artifacts = _build(tmp_path)
    paths[1].write_text(paths[1].read_text() + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="JSONL hash mismatch"):
        reconcile_safety_batches(paths)


def test_row_count_mismatch_is_rejected_even_with_updated_hash(tmp_path: Path) -> None:
    paths, _artifacts = _build(tmp_path)
    paths[1].write_text("", encoding="utf-8")
    manifest_path = paths[1].parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["candidate_safety_jsonl"]["sha256"] = sha256_file(paths[1])
    _write_json(manifest_path, manifest)
    with pytest.raises(SystemExit, match="row count mismatch"):
        reconcile_safety_batches(paths)


def test_missing_batch_manifest_is_rejected(tmp_path: Path) -> None:
    paths, _artifacts = _build(tmp_path)
    (paths[0].parent / "manifest.json").unlink()
    with pytest.raises(SystemExit, match="manifest is missing"):
        reconcile_safety_batches(paths)


def test_reconciliation_manifest_row_count_is_respected(tmp_path: Path) -> None:
    paths, _artifacts = _build(tmp_path)
    rows = [json.loads(line) for line in paths[0].read_text().splitlines()]
    paths[0].write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    manifest_path = paths[0].parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        schema=SCHEMA + ".reconciliation",
        selected_unique_compound_count=1,
        input_row_count=1,
        artifacts={
            "candidate_safety_jsonl": {
                "path": str(paths[0]),
                "sha256": sha256_file(paths[0]),
            }
        },
    )
    _write_json(manifest_path, manifest)
    reconciled, _comparisons = reconcile_safety_batches(paths)
    assert len(reconciled) == 1


def test_real_185_row_batches_still_reconcile_to_182_with_three_duplicates() -> None:
    base = ROOT / "results/target_first_20260915/safety"
    main = base / "candidate_safety.jsonl"
    additional = base / "additional" / "candidate_safety.jsonl"
    if not main.is_file() or not additional.is_file():
        pytest.skip("target-first safety batches are not present in this checkout")
    rows, comparisons = reconcile_safety_batches([main, additional])
    assert len(rows) == 182
    assert len(comparisons) == 3
