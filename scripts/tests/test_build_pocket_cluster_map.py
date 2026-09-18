"""Regression tests for provenance-bound Foldseek pocket cluster maps."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "eval" / "build_pocket_cluster_map.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _build_fixture(tmp_path: Path) -> None:
    targets = tmp_path / "targets.csv"
    _write_csv(
        targets,
        [
            {"uniprot": "P1", "target_cluster_30": "P1", "target_cluster_50": "P1"},
            {"uniprot": "P2", "target_cluster_30": "P1", "target_cluster_50": "P2"},
            {"uniprot": "P3", "target_cluster_30": "P3", "target_cluster_50": "P3"},
        ],
        ["uniprot", "target_cluster_30", "target_cluster_50"],
    )
    target_manifest = {
        "schema_version": "skinscout.target-cluster-map.v2",
        "artifact": {
            "path": str(targets.resolve()),
            "sha256": _sha256(targets),
            "rows": 3,
        },
    }
    (tmp_path / "targets.manifest.json").write_text(json.dumps(target_manifest) + "\n")

    fragment_root = tmp_path / "fragment_root"
    (fragment_root / "fragments").mkdir(parents=True, exist_ok=True)
    for uid in ("P1", "P2"):
        (fragment_root / "fragments" / f"{uid}_p2rank_top1_context2.pdb").write_text(
            f"HEADER {uid}\n"
        )
    index_rows = []
    for uid in ("P1", "P2"):
        rel = f"fragments/{uid}_p2rank_top1_context2.pdb"
        index_rows.append(
            {
                "uniprot": uid,
                "pocket_rank": "1",
                "pocket_score": "1.0",
                "seed_residue_count": "1",
                "fragment_residue_count": "12",
                "source_pdb_sha256": "a" * 64,
                "source_prediction_sha256": "b" * 64,
                "fragment_path": rel,
                "fragment_sha256": _sha256(fragment_root / rel),
            }
        )
    _write_csv(
        tmp_path / "fragments.csv",
        index_rows,
        [
            "uniprot",
            "pocket_rank",
            "pocket_score",
            "seed_residue_count",
            "fragment_residue_count",
            "source_pdb_sha256",
            "source_prediction_sha256",
            "fragment_path",
            "fragment_sha256",
        ],
    )
    _write_csv(
        tmp_path / "fragment_exclusions.csv",
        [{"uniprot": "P3", "reason": "missing_structure_and_pocket"}],
        ["uniprot", "reason"],
    )
    fragment_manifest = {
        "schema_version": "skinscout.pocket-fragment-universe.v1",
        "target_manifest_provenance": {
            "path": str((tmp_path / "targets.manifest.json").resolve()),
            "sha256": _sha256(tmp_path / "targets.manifest.json"),
            "schema_version": "skinscout.target-cluster-map.v2",
            "artifact": {
                "path": str(targets.resolve()),
                "sha256": _sha256(targets),
                "rows": 3,
            },
        },
        "counts": {"targets": 3, "accepted": 2, "excluded": 1},
        "artifacts": {
            "out_dir": str(fragment_root.resolve()),
            "out_dir_tree_sha256": _tree_digest(fragment_root),
            "index": {
                "path": str((tmp_path / "fragments.csv").resolve()),
                "sha256": _sha256(tmp_path / "fragments.csv"),
                "rows": 2,
            },
            "exclusions": {
                "path": str((tmp_path / "fragment_exclusions.csv").resolve()),
                "sha256": _sha256(tmp_path / "fragment_exclusions.csv"),
                "rows": 1,
            },
        },
    }
    (tmp_path / "fragments.manifest.json").write_text(json.dumps(fragment_manifest) + "\n")
    for threshold in ("40", "50", "60"):
        (tmp_path / f"tm{threshold}.tsv").write_text(
            "P1_p2rank_top1_context2\tP1_p2rank_top1_context2\n"
            "P1_p2rank_top1_context2\tP2_p2rank_top1_context2\n"
        )


def _run(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--fragment-index",
            str(tmp_path / "fragments.csv"),
            "--fragment-exclusions",
            str(tmp_path / "fragment_exclusions.csv"),
            "--fragment-manifest",
            str(tmp_path / "fragments.manifest.json"),
            "--target-clusters",
            str(tmp_path / "targets.csv"),
            "--target-cluster-manifest",
            str(tmp_path / "targets.manifest.json"),
            "--foldseek-tm40-tsv",
            str(tmp_path / "tm40.tsv"),
            "--foldseek-tm50-tsv",
            str(tmp_path / "tm50.tsv"),
            "--foldseek-tm60-tsv",
            str(tmp_path / "tm60.tsv"),
            "--foldseek-version",
            "9.427df8a",
            "--foldseek-command-tm40",
            "foldseek easy-cluster fragments pocket_tm40 tmp --tmscore-threshold 0.4",
            "--foldseek-command-tm50",
            "foldseek easy-cluster fragments pocket_tm50 tmp --tmscore-threshold 0.5",
            "--foldseek-command-tm60",
            "foldseek easy-cluster fragments pocket_tm60 tmp --tmscore-threshold 0.6",
            "--expected-target-count",
            "3",
            "--out-csv",
            str(tmp_path / "pocket_clusters.csv"),
            "--out-manifest",
            str(tmp_path / "pocket_clusters.manifest.json"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_build_pocket_cluster_map_is_deterministic_and_provenance_bound(tmp_path: Path) -> None:
    _build_fixture(tmp_path)

    first = _run(tmp_path)
    first_csv = (tmp_path / "pocket_clusters.csv").read_text()
    second = _run(tmp_path)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert (tmp_path / "pocket_clusters.csv").read_text() == first_csv
    assert _read_csv(tmp_path / "pocket_clusters.csv") == [
        {
            "uniprot": "P1",
            "pocket_available": "true",
            "exclusion_reason": "",
            "pocket_cluster_tm40": "P1",
            "pocket_cluster_tm50": "P1",
            "pocket_cluster_tm60": "P1",
        },
        {
            "uniprot": "P2",
            "pocket_available": "true",
            "exclusion_reason": "",
            "pocket_cluster_tm40": "P1",
            "pocket_cluster_tm50": "P1",
            "pocket_cluster_tm60": "P1",
        },
        {
            "uniprot": "P3",
            "pocket_available": "false",
            "exclusion_reason": "missing_structure_and_pocket",
            "pocket_cluster_tm40": "",
            "pocket_cluster_tm50": "",
            "pocket_cluster_tm60": "",
        },
    ]
    manifest = json.loads((tmp_path / "pocket_clusters.manifest.json").read_text())
    assert manifest["schema_version"] == "skinscout.pocket-cluster-map.v1"
    assert manifest["parameters"]["foldseek_version"] == "9.427df8a"
    assert manifest["parameters"]["main_tm_threshold"] == 0.5
    assert manifest["parameters"]["sensitivity_tm_thresholds"] == [0.4, 0.6]
    assert "whole-protein" in manifest["definition"]
    assert manifest["artifact"]["sha256"] == _sha256(tmp_path / "pocket_clusters.csv")
    assert manifest["counts"] == {
        "targets": 3,
        "pocket_available": 2,
        "pocket_unavailable": 1,
        "clusters": {"tm40": 1, "tm50": 1, "tm60": 1},
    }


def test_provenance_mismatch_fails_closed_and_removes_stale_outputs(tmp_path: Path) -> None:
    _build_fixture(tmp_path)
    payload = json.loads((tmp_path / "fragments.manifest.json").read_text())
    payload["artifacts"]["index"]["sha256"] = "0" * 64
    (tmp_path / "fragments.manifest.json").write_text(json.dumps(payload) + "\n")
    (tmp_path / "pocket_clusters.csv").write_text("stale\n")
    (tmp_path / "pocket_clusters.manifest.json").write_text("stale\n")

    result = _run(tmp_path)

    assert result.returncode != 0
    assert "pocket fragment index CSV sha256 does not match manifest" in result.stderr
    assert not (tmp_path / "pocket_clusters.csv").exists()
    assert not (tmp_path / "pocket_clusters.manifest.json").exists()


def test_missing_duplicate_and_unknown_foldseek_members_fail_closed(tmp_path: Path) -> None:
    _build_fixture(tmp_path)
    (tmp_path / "tm40.tsv").write_text("P1_p2rank_top1_context2\tP1_p2rank_top1_context2\n")
    missing = _run(tmp_path)
    assert missing.returncode != 0
    assert "is incomplete" in missing.stderr

    _build_fixture(tmp_path)
    (tmp_path / "tm40.tsv").write_text(
        "P1_p2rank_top1_context2\tP1_p2rank_top1_context2\n"
        "P2_p2rank_top1_context2\tP1_p2rank_top1_context2\n"
    )
    duplicate = _run(tmp_path)
    assert duplicate.returncode != 0
    assert "more than once" in duplicate.stderr

    _build_fixture(tmp_path)
    (tmp_path / "tm40.tsv").write_text(
        "P1_p2rank_top1_context2\tP1_p2rank_top1_context2\n"
        "P1_p2rank_top1_context2\tPX_p2rank_top1_context2\n"
    )
    unknown = _run(tmp_path)
    assert unknown.returncode != 0
    assert "unknown Foldseek identifier" in unknown.stderr


def test_accepted_excluded_partition_must_match_target_universe(tmp_path: Path) -> None:
    _build_fixture(tmp_path)
    _write_csv(tmp_path / "fragment_exclusions.csv", [], ["uniprot", "reason"])
    payload = json.loads((tmp_path / "fragments.manifest.json").read_text())
    payload["artifacts"]["exclusions"]["sha256"] = _sha256(tmp_path / "fragment_exclusions.csv")
    payload["artifacts"]["exclusions"]["rows"] = 0
    payload["counts"]["excluded"] = 0
    (tmp_path / "fragments.manifest.json").write_text(json.dumps(payload) + "\n")

    result = _run(tmp_path)

    assert result.returncode != 0
    assert "exactly partition target universe" in result.stderr


def test_unavailable_targets_have_blank_cluster_columns(tmp_path: Path) -> None:
    _build_fixture(tmp_path)

    result = _run(tmp_path)

    assert result.returncode == 0, result.stderr
    unavailable = [
        row
        for row in _read_csv(tmp_path / "pocket_clusters.csv")
        if row["pocket_available"] == "false"
    ][0]
    assert unavailable["uniprot"] == "P3"
    assert unavailable["pocket_cluster_tm40"] == ""
    assert unavailable["pocket_cluster_tm50"] == ""
    assert unavailable["pocket_cluster_tm60"] == ""
