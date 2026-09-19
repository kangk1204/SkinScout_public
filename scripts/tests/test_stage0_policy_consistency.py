"""C07: producer, readiness and final verifier must share one Stage 0 policy.

The producer records expected/success/failed receptors (with reasons and input
hashes) in a manifest and enforces ``>= min_success_fraction`` instead of a
perfect set. These tests pin the non-destructive policy: partial sets are
acceptable when the manifest proves the configured gate was met, and a touch
file alone is never enough.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
TESTS = Path(__file__).resolve().parent
for entry in (str(SCRIPTS), str(TESTS)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import data_readiness
import stage0_verify
from data_readiness import (
    apply_config_overrides,
    check_required_artifact,
    data_readiness_payload,
    load_workflow_config,
    required_data_artifacts,
    resolve_stage0_paths,
)
from stage0_manifest import (
    MANIFEST_FILENAME,
    build_manifest,
    validate_manifest,
    write_manifest,
)

DIGEST = "0" * 64


def _record(target: str, status: str, reason: str | None = None) -> dict:
    return {
        "uniprot": target,
        "status": status,
        "method": "meeko" if status == "ok" else None,
        "reason": reason,
        "clean_pdb_sha256": DIGEST,
        "pocket_json_sha256": DIGEST,
    }


def _write_manifest(
    pdbqt_dir: Path,
    *,
    records: list[dict],
    fraction: float = 0.8,
    count: int = 1,
    scope: str = "full",
) -> Path:
    pdbqt_dir.mkdir(parents=True, exist_ok=True)
    document = build_manifest(
        scope=scope,
        policy={"min_success_fraction": fraction, "min_success_count": count},
        records=records,
    )
    path = pdbqt_dir / MANIFEST_FILENAME
    write_manifest(path, document)
    return path


def _write_pdbqt_files(pdbqt_dir: Path, targets: list[str]) -> None:
    pdbqt_dir.mkdir(parents=True, exist_ok=True)
    for target in targets:
        (pdbqt_dir / f"{target}.pdbqt").write_text("REMARK ok\n")


def test_producer_manifest_counts_the_real_gate(tmp_path: Path) -> None:
    pdbqt_dir = tmp_path / "pdbqt"
    records = [_record(f"P{idx:05d}", "ok") for idx in range(9)]
    records.append(_record("P00009", "failed", "meeko: no polar hydrogens"))
    _write_pdbqt_files(pdbqt_dir, [record["uniprot"] for record in records if record["status"] == "ok"])

    manifest = _write_manifest(pdbqt_dir, records=records)
    document = json.loads(manifest.read_text())

    assert document["status"] == "ok"
    assert document["counts"] == {"expected": 10, "success": 9, "failed": 1}
    assert document["targets"][-1]["reason"] == "meeko: no polar hydrogens"


def test_one_missing_receptor_is_consistent_across_producer_ready_and_verifier(
    tmp_path: Path,
) -> None:
    pdbqt_dir = tmp_path / "pdbqt"
    records = [_record(f"P{idx:05d}", "ok") for idx in range(9)]
    records.append(_record("P00009", "failed", "meeko 엄격 실패; -a 재시도도 실패"))
    _write_pdbqt_files(pdbqt_dir, [record["uniprot"] for record in records if record["status"] == "ok"])
    manifest = _write_manifest(pdbqt_dir, records=records)

    # producer policy accepts a 90% set
    assert json.loads(manifest.read_text())["status"] == "ok"

    # readiness reads the manifest, not the touch file
    readiness = check_required_artifact("PDBQT receptor marker", manifest)
    assert readiness.status == "present"
    assert "expected=10 success=9 failed=1" in readiness.detail
    assert "P00009" in readiness.detail
    assert "-a 재시도도 실패" in readiness.detail

    # final verifier applies the same policy
    check = stage0_verify.chk_pdbqt_roundtrip(
        pdbqt_dir,
        manifest_path=manifest,
        expected_policy={"min_success_fraction": 0.8, "min_success_count": 1},
    )
    assert check.ok, check.detail
    assert "P00009" in check.detail

    (pdbqt_dir / ".pdbqt_complete").write_text("")
    assert check_required_artifact("PDBQT receptor marker", manifest).status == "present"


def test_touch_file_alone_cannot_satisfy_readiness(tmp_path: Path) -> None:
    pdbqt_dir = tmp_path / "pdbqt"
    pdbqt_dir.mkdir()
    (pdbqt_dir / ".pdbqt_complete").write_text("")
    _write_pdbqt_files(pdbqt_dir, ["P00000"])
    manifest = pdbqt_dir / MANIFEST_FILENAME
    assert not manifest.exists()

    readiness = check_required_artifact("PDBQT receptor marker", manifest)
    assert readiness.status == "missing"

    check = stage0_verify.chk_pdbqt_roundtrip(
        pdbqt_dir,
        manifest_path=manifest,
        expected_policy={"min_success_fraction": 0.8, "min_success_count": 1},
    )
    assert not check.ok
    assert "missing" in check.detail


def test_success_target_without_a_file_fails_verifier_and_readiness(tmp_path: Path) -> None:
    pdbqt_dir = tmp_path / "pdbqt"
    manifest = _write_manifest(
        pdbqt_dir,
        records=[_record("P00000", "ok"), _record("P00001", "failed", "meeko")],
    )
    # A failed target keeps the directory non-empty so the manifest check runs.
    (pdbqt_dir / "P00001.pdbqt").write_text("REMARK ok\n")

    check = stage0_verify.chk_pdbqt_roundtrip(
        pdbqt_dir,
        manifest_path=manifest,
        expected_policy={"min_success_fraction": 0.8, "min_success_count": 1},
    )
    assert not check.ok
    assert "P00000" in check.detail

    readiness = check_required_artifact("PDBQT receptor marker", manifest)
    assert readiness.status == "invalid"


def test_failed_target_without_a_reason_is_rejected(tmp_path: Path) -> None:
    pdbqt_dir = tmp_path / "pdbqt"
    manifest = _write_manifest(
        pdbqt_dir,
        records=[_record("P00000", "ok"), _record("P00001", "failed", None)],
    )
    payload = json.loads(manifest.read_text())
    _index, errors = validate_manifest(payload, pdbqt_dir=pdbqt_dir)
    assert any("reason" in error for error in errors)


def test_corrupt_manifest_is_invalid_and_not_buildable(tmp_path: Path) -> None:
    manifest = tmp_path / MANIFEST_FILENAME
    manifest.write_text("{not-json\n")

    check = check_required_artifact("PDBQT receptor marker", manifest)
    assert check.status == "invalid"

    entry = data_readiness.failed_check_entry(check)
    assert not data_readiness.failed_entries_are_stage0_buildable([entry])


def test_weaker_policy_than_the_configured_gate_is_rejected(tmp_path: Path) -> None:
    pdbqt_dir = tmp_path / "pdbqt"
    manifest = _write_manifest(
        pdbqt_dir,
        records=[_record("P00000", "ok"), _record("P00001", "failed", "meeko")],
        fraction=0.5,
    )
    _write_pdbqt_files(pdbqt_dir, ["P00000"])

    check = stage0_verify.chk_pdbqt_roundtrip(
        pdbqt_dir,
        manifest_path=manifest,
        expected_policy={"min_success_fraction": 0.8, "min_success_count": 1},
    )
    assert not check.ok
    assert "weaker than the configured gate" in check.detail


def test_restricted_scope_manifest_cannot_claim_a_full_run(tmp_path: Path) -> None:
    pdbqt_dir = tmp_path / "pdbqt"
    manifest = _write_manifest(
        pdbqt_dir,
        records=[_record("P00000", "ok")],
        scope="restricted",
    )
    _write_pdbqt_files(pdbqt_dir, ["P00000"])

    check = stage0_verify.chk_pdbqt_roundtrip(
        pdbqt_dir,
        manifest_path=manifest,
        expected_policy={"min_success_fraction": 0.8, "min_success_count": 1},
    )
    assert not check.ok
    assert "restricted" in check.detail


def test_manifest_expected_set_owns_the_verdict(tmp_path: Path) -> None:
    pdbqt_dir = tmp_path / "pdbqt"
    records = [_record(f"P{idx:05d}", "ok") for idx in range(5)]
    records[1] = _record("P00001", "failed", "no measurable pocket extent")
    manifest = _write_manifest(pdbqt_dir, records=records)
    _write_pdbqt_files(
        pdbqt_dir,
        [record["uniprot"] for record in records if record["status"] == "ok"],
    )

    check = stage0_verify.chk_pdbqt_roundtrip(
        pdbqt_dir,
        expected_targets={"P00000", "P00001", "P00002", "P00003", "P00004", "P09999"},
        manifest_path=manifest,
        expected_policy={"min_success_fraction": 0.8, "min_success_count": 1},
    )

    assert check.ok, check.detail
    assert "P09999" in check.detail


def test_missing_fasta_fails_readiness_and_the_final_verifier(tmp_path: Path) -> None:
    config = load_workflow_config()
    apply_config_overrides(
        config,
        [
            f"paths.mmseqs={tmp_path / 'mmseqs'}",
            f"paths.uniprot_human_fasta={tmp_path / 'mmseqs' / 'human_canonical.fasta'}",
        ],
    )

    artifacts = {
        label: path
        for label, path in required_data_artifacts(
            "target-id",
            "fast",
            config=config,
            root=tmp_path,
        )
    }
    assert "canonical human FASTA" in artifacts
    assert artifacts["canonical human FASTA"] == (
        tmp_path / "mmseqs" / "human_canonical.fasta"
    )

    payload = data_readiness_payload(
        "target-id",
        "fast",
        False,
        config=config,
        root=tmp_path,
    )
    assert payload["status"] == "failed"
    assert any(
        check["label"] == "canonical human FASTA" and check["status"] == "missing"
        for check in payload["checks"]
    )

    checks = stage0_verify.collect_checks(
        tmp_path,
        claim_quality=True,
        check_stage0_flag=False,
        min_cosing_rows=1,
        min_drug_rows=1,
        min_skin_score_rows=1,
        min_kg_gene_edges=1,
        paths=resolve_stage0_paths(config, root=tmp_path),
    )
    by_name = {check.name: check for check in checks}
    assert not by_name["claim_canonical_human_sequences"].ok
    assert "missing" in by_name["claim_canonical_human_sequences"].detail


def test_producer_subprocess_writes_a_failed_manifest_even_without_meeko(
    tmp_path: Path,
) -> None:
    from test_stage0_meeko_prep_fail_closed import run_meeko_prep

    res = run_meeko_prep(
        tmp_path,
        ["--min-success-count", "0", "--min-success-fraction", "0.0"],
    )

    assert res.returncode == 0, res.stderr
    manifest = tmp_path / "pdbqt" / MANIFEST_FILENAME
    document = json.loads(manifest.read_text())
    assert document["schema_version"] == "skinscout.stage0-receptor-pdbqt.v1"
    assert document["scope"] == "full"
    assert document["policy"] == {"min_success_fraction": 0.0, "min_success_count": 0}
    assert document["counts"] == {"expected": 1, "success": 0, "failed": 1}
    record = document["targets"][0]
    assert record["status"] == "failed"
    assert record["reason"]
    assert record["clean_pdb_sha256"]
    assert record["pocket_json_sha256"]


def test_producer_strict_gate_blocks_a_failed_manifest(tmp_path: Path) -> None:
    from test_stage0_meeko_prep_fail_closed import run_meeko_prep

    res = run_meeko_prep(tmp_path)

    assert res.returncode != 0
    document = json.loads((tmp_path / "pdbqt" / MANIFEST_FILENAME).read_text())
    assert document["status"] == "failed"
    assert document["counts"]["failed"] == 1
