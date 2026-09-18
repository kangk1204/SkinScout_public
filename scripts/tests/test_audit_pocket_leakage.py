"""Regression tests for the provenance-bound pocket leakage sidecar audit."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "eval" / "audit_pocket_leakage.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _bench_row(uid: str, split: str, *, evidence_id: str | None = None) -> dict[str, object]:
    return {
        "benchmark_id": f"bench-{split}-{uid}",
        "evidence_id": evidence_id or f"ev-{split}-{uid}",
        "split": split,
        "uniprot": uid,
        "target_cluster_30": f"seq30-{uid}",
        "target_cluster_50": f"seq50-{uid}",
    }


def _write_target_map(tmp_path: Path, targets: list[str]) -> tuple[Path, Path]:
    target_csv = tmp_path / "target_clusters.csv"
    _write_csv(
        target_csv,
        [
            {
                "uniprot": uid,
                "target_cluster_30": f"seq30-{uid}",
                "target_cluster_50": f"seq50-{uid}",
            }
            for uid in targets
        ],
        ["uniprot", "target_cluster_30", "target_cluster_50"],
    )
    target_manifest = tmp_path / "target_clusters.manifest.json"
    target_manifest.write_text(
        json.dumps(
            {
                        "schema_version": "skinscout.screenable-target-cluster-map.v2",
                "artifact": {
                    "path": str(target_csv.resolve()),
                    "sha256": _sha256(target_csv),
                    "rows": len(targets),
                },
            }
        )
        + "\n"
    )
    return target_csv, target_manifest


def _write_pocket_map(
    tmp_path: Path,
    target_csv: Path,
    target_manifest: Path,
    targets: list[str],
    *,
    overlap: bool = False,
    unavailable: bool = False,
) -> tuple[Path, Path]:
    cluster_by_uid = {
        "P1": "pocket-a",
        "P2": "pocket-b",
        "P3": "pocket-a" if overlap else "pocket-c",
        "P4": "pocket-d",
        "P5": "",
    }
    rows = []
    for uid in targets:
        is_unavailable = unavailable and uid == "P5"
        base = cluster_by_uid[uid]
        rows.append(
            {
                "uniprot": uid,
                "pocket_available": "false" if is_unavailable else "true",
                "exclusion_reason": "missing_structure_and_pocket" if is_unavailable else "",
                "pocket_cluster_tm40": "" if is_unavailable else f"{base}-tm40",
                "pocket_cluster_tm50": "" if is_unavailable else f"{base}-tm50",
                "pocket_cluster_tm60": "" if is_unavailable else f"{base}-tm60",
            }
        )
    pocket_csv = tmp_path / "pocket_clusters.csv"
    _write_csv(
        pocket_csv,
        rows,
        [
            "uniprot",
            "pocket_available",
            "exclusion_reason",
            "pocket_cluster_tm40",
            "pocket_cluster_tm50",
            "pocket_cluster_tm60",
        ],
    )
    pocket_manifest = tmp_path / "pocket_clusters.manifest.json"
    pocket_manifest.write_text(
        json.dumps(
            {
                "schema_version": "skinscout.pocket-cluster-map.v1",
                "inputs": {
                    "target_clusters": {
                        "path": str(target_csv.resolve()),
                        "sha256": _sha256(target_csv),
                        "rows": len(targets),
                    },
                    "target_cluster_manifest": {
                        "path": str(target_manifest.resolve()),
                        "sha256": _sha256(target_manifest),
                "schema_version": "skinscout.screenable-target-cluster-map.v2",
                    },
                },
                "artifact": {
                    "path": str(pocket_csv.resolve()),
                    "sha256": _sha256(pocket_csv),
                    "rows": len(rows),
                },
            }
        )
        + "\n"
    )
    return pocket_csv, pocket_manifest


def _write_benchmark(
    tmp_path: Path,
    target_csv: Path,
    target_manifest: Path,
    *,
    unavailable: bool = False,
) -> Path:
    train = pd.DataFrame([_bench_row("P1", "train")])
    dev = pd.DataFrame([_bench_row("P2", "dev")])
    test_rows = [_bench_row("P3", "test")]
    if unavailable:
        test_rows.append(_bench_row("P5", "test"))
    test = pd.DataFrame(test_rows)
    dual = pd.DataFrame(
        [
            {
                **_bench_row("P4", "test", evidence_id="ev-dual-P4"),
                "evaluation_view": "dual_cold",
            }
        ]
    )
    for name, frame in {
        "train": train,
        "dev": dev,
        "test": test,
        "dual_cold": dual,
    }.items():
        frame.to_parquet(tmp_path / f"{name}.parquet", index=False)
    manifest = {
        "schema_version": "activity_benchmark.v1",
        "output_sha256": {
            "train.parquet": _sha256(tmp_path / "train.parquet"),
            "dev.parquet": _sha256(tmp_path / "dev.parquet"),
            "test.parquet": _sha256(tmp_path / "test.parquet"),
            "dual_cold.parquet": _sha256(tmp_path / "dual_cold.parquet"),
        },
        "splits": {"counts": {"train": len(train), "dev": len(dev), "test": len(test)}},
        "evaluation_views": {"counts": {"dual_cold": len(dual)}},
        "target_cluster_policy": {
            "required_input": {
                "path": str(target_csv.resolve()),
                "sha256": _sha256(target_csv),
                "rows": 5,
                "manifest_path": str(target_manifest.resolve()),
                "manifest_sha256": _sha256(target_manifest),
            }
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest) + "\n")
    return manifest_path


def _fixture(
    tmp_path: Path,
    *,
    overlap: bool = False,
    unavailable: bool = False,
) -> tuple[Path, Path, Path, Path]:
    targets = ["P1", "P2", "P3", "P4", "P5"]
    target_csv, target_manifest = _write_target_map(tmp_path, targets)
    pocket_csv, pocket_manifest = _write_pocket_map(
        tmp_path,
        target_csv,
        target_manifest,
        targets,
        overlap=overlap,
        unavailable=unavailable,
    )
    benchmark_manifest = _write_benchmark(
        tmp_path,
        target_csv,
        target_manifest,
        unavailable=unavailable,
    )
    return benchmark_manifest, target_csv, target_manifest, pocket_csv, pocket_manifest


def _run(
    tmp_path: Path,
    benchmark_manifest: Path,
    target_csv: Path,
    target_manifest: Path,
    pocket_csv: Path,
    pocket_manifest: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--benchmark-manifest",
            str(benchmark_manifest),
            "--train-parquet",
            str(tmp_path / "train.parquet"),
            "--dev-parquet",
            str(tmp_path / "dev.parquet"),
            "--test-parquet",
            str(tmp_path / "test.parquet"),
            "--dual-cold-parquet",
            str(tmp_path / "dual_cold.parquet"),
            "--target-clusters",
            str(target_csv),
            "--target-cluster-manifest",
            str(target_manifest),
            "--pocket-clusters",
            str(pocket_csv),
            "--pocket-cluster-manifest",
            str(pocket_manifest),
            "--out-json",
            str(tmp_path / "audit.json"),
            "--out-csv",
            str(tmp_path / "audit_rows.csv"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_clean_pocket_audit_reports_no_leakage_and_deterministic_outputs(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    first = _run(tmp_path, *fixture)
    first_json = (tmp_path / "audit.json").read_text()
    second = _run(tmp_path, *fixture)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert (tmp_path / "audit.json").read_text() == first_json
    payload = json.loads(first_json)
    assert payload["schema_version"] == "skinscout.pocket-leakage-audit.v1"
    assert payload["by_split"]["test"]["pocket_cluster_leakage"]["tm50"]["train"]["seen_count"] == 0
    assert payload["by_split"]["test"]["pocket_cluster_leakage"]["tm50"]["prior_splits"]["pocket_cold_count"] == 1
    rows = _read_csv(tmp_path / "audit_rows.csv")
    assert [row["audit_split"] for row in rows] == ["dev", "test", "dual_cold"]
    assert rows[1]["sequence_cold_30_from_prior_splits"] == "true"
    assert rows[1]["pocket_cold_tm50_from_prior_splits"] == "true"


def test_pocket_overlap_is_reported_separately_from_sequence_cold(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, overlap=True)

    result = _run(tmp_path, *fixture)

    assert result.returncode == 0, result.stderr
    payload = json.loads((tmp_path / "audit.json").read_text())
    test_tm50 = payload["by_split"]["test"]["pocket_cluster_leakage"]["tm50"]
    assert test_tm50["train"]["seen_count"] == 1
    assert test_tm50["prior_splits"]["seen_count"] == 1
    rows = _read_csv(tmp_path / "audit_rows.csv")
    test_row = next(row for row in rows if row["audit_split"] == "test")
    assert test_row["sequence_cold_30_from_prior_splits"] == "true"
    assert test_row["pocket_tm50_seen_in_prior_splits"] == "true"
    assert test_row["pocket_cold_tm50_from_prior_splits"] == "false"


def test_unavailable_pockets_are_counted_but_not_pocket_cold(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, unavailable=True)

    result = _run(tmp_path, *fixture)

    assert result.returncode == 0, result.stderr
    payload = json.loads((tmp_path / "audit.json").read_text())
    summary = payload["by_split"]["test"]
    assert summary["rows"] == 2
    assert summary["pocket_unavailable_rows"] == 1
    assert summary["pocket_cluster_leakage"]["tm50"]["prior_splits"]["unavailable_pocket_count"] == 1
    rows = _read_csv(tmp_path / "audit_rows.csv")
    unavailable = next(row for row in rows if row["uniprot"] == "P5")
    assert unavailable["pocket_available"] == "false"
    assert unavailable["pocket_cold_tm50_from_prior_splits"] == "false"


def test_provenance_mismatch_fails_closed_and_removes_stale_outputs(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    benchmark_manifest, target_csv, target_manifest, pocket_csv, pocket_manifest = fixture
    payload = json.loads(benchmark_manifest.read_text())
    payload["output_sha256"]["train.parquet"] = "0" * 64
    benchmark_manifest.write_text(json.dumps(payload) + "\n")
    (tmp_path / "audit.json").write_text("stale\n")
    (tmp_path / "audit_rows.csv").write_text("stale\n")

    result = _run(
        tmp_path,
        benchmark_manifest,
        target_csv,
        target_manifest,
        pocket_csv,
        pocket_manifest,
    )

    assert result.returncode != 0
    assert "activity benchmark train.parquet sha256 does not match manifest" in result.stderr
    assert not (tmp_path / "audit.json").exists()
    assert not (tmp_path / "audit_rows.csv").exists()
