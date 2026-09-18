"""Regression tests for provenance-bound P2Rank pocket fragments."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/build_pocket_fragments.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pdb_atom(serial: int, atom: str, resname: str, chain: str, resseq: int, x: float) -> str:
    return (
        f"ATOM  {serial:5d} {atom:^4s} {resname:>3s} {chain}{resseq:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{20.0:6.2f}           {atom[0]:>2s}\n"
    )


def _write_pdb(path: Path, residues: int = 16) -> None:
    names = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE"]
    lines: list[str] = []
    serial = 1
    for idx in range(1, residues + 1):
        resname = names[(idx - 1) % len(names)]
        for atom in ("N", "CA", "C"):
            lines.append(_pdb_atom(serial, atom, resname, "A", idx, float(idx)))
            serial += 1
    lines.append("TER\nEND\n")
    path.write_text("".join(lines))


def _write_prediction(path: Path, rows: list[str] | None = None) -> None:
    path.write_text(
        " name, rank, score, residue_ids\n"
        + "".join(
            ["p1, 1, 7.5, A_8 A_9\n"] if rows is None else rows
        )
    )


def _write_targets_and_manifest(tmp_path: Path, targets: list[str]) -> tuple[Path, Path]:
    targets_csv = tmp_path / "targets.csv"
    pd.DataFrame(
        {
            "uniprot": targets,
            "target_cluster_30": targets,
            "target_cluster_50": targets,
        }
    ).to_csv(targets_csv, index=False)
    manifest = tmp_path / "target_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "skinscout.target-cluster-map.v2",
                "artifact": {
                    "path": str(targets_csv),
                    "sha256": _sha256(targets_csv),
                    "rows": len(targets),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return targets_csv, manifest


def _run(
    tmp_path: Path,
    targets: list[str] | None = None,
    extra: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    targets_csv, manifest = _write_targets_and_manifest(tmp_path, targets or ["P12345"])
    params = tmp_path / "p2rank.params"
    params.write_text("conservation=false\n")
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--targets-csv",
            str(targets_csv),
            "--target-cluster-manifest",
            str(manifest),
            "--pdb-dir",
            str(tmp_path / "pdb"),
            "--pocket-dir",
            str(tmp_path / "pockets"),
            "--p2rank-params",
            str(params),
            "--p2rank-version",
            "2.5",
            "--context-residues",
            "2",
            "--min-fragment-residues",
            "5",
            "--out-dir",
            str(tmp_path / "fragments_out"),
            "--out-index",
            str(tmp_path / "index.csv"),
            "--out-exclusions",
            str(tmp_path / "exclusions.csv"),
            "--out-manifest",
            str(tmp_path / "manifest.json"),
            *(extra or []),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_context_fragment_and_hashes_are_deterministic(tmp_path: Path) -> None:
    pdb_dir = tmp_path / "pdb"
    pocket_dir = tmp_path / "pockets"
    pdb_dir.mkdir()
    pocket_dir.mkdir()
    _write_pdb(pdb_dir / "P12345_clean.pdb")
    _write_prediction(pocket_dir / "P12345_clean.pdb_predictions.csv")

    first = _run(tmp_path)
    assert first.returncode == 0, first.stderr
    index = pd.read_csv(tmp_path / "index.csv")
    assert index.to_dict("records")[0]["fragment_residue_count"] == 6
    assert index.to_dict("records")[0]["seed_residue_count"] == 2
    assert index.to_dict("records")[0]["source_pdb_sha256"] == _sha256(pdb_dir / "P12345_clean.pdb")
    fragment = tmp_path / "fragments_out" / index.loc[0, "fragment_path"]
    assert fragment.exists()
    assert index.loc[0, "fragment_sha256"] == _sha256(fragment)
    text = fragment.read_text()
    assert " A   6" in text
    assert " A  11" in text
    assert " A   5" not in text
    first_sha = index.loc[0, "fragment_sha256"]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["schema_version"] == "skinscout.pocket-fragment-universe.v1"
    assert manifest["counts"] == {"accepted": 1, "excluded": 0, "targets": 1}
    assert "not whole-protein similarity" in manifest["definition"]

    second = _run(tmp_path)
    assert second.returncode == 0, second.stderr
    second_index = pd.read_csv(tmp_path / "index.csv")
    assert second_index.loc[0, "fragment_sha256"] == first_sha


def test_both_missing_is_audited_exclusion(tmp_path: Path) -> None:
    (tmp_path / "pdb").mkdir()
    (tmp_path / "pockets").mkdir()

    result = _run(tmp_path)

    assert result.returncode == 0, result.stderr
    assert pd.read_csv(tmp_path / "index.csv").empty
    assert pd.read_csv(tmp_path / "exclusions.csv").to_dict("records") == [
        {"uniprot": "P12345", "reason": "missing_structure_and_pocket"}
    ]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["counts"]["excluded"] == 1


def test_one_missing_fails_closed_and_cleans_stale_outputs(tmp_path: Path) -> None:
    pdb_dir = tmp_path / "pdb"
    (tmp_path / "pockets").mkdir()
    pdb_dir.mkdir()
    _write_pdb(pdb_dir / "P12345_clean.pdb")
    (tmp_path / "fragments_out").mkdir()
    (tmp_path / "fragments_out" / "stale.pdb").write_text("stale\n")
    (tmp_path / "index.csv").write_text("stale\n")
    (tmp_path / "exclusions.csv").write_text("stale\n")
    (tmp_path / "manifest.json").write_text("stale\n")

    result = _run(tmp_path)

    assert result.returncode != 0
    assert "refusing partial input" in result.stderr
    assert not (tmp_path / "fragments_out").exists()
    assert not (tmp_path / "index.csv").exists()
    assert not (tmp_path / "exclusions.csv").exists()
    assert not (tmp_path / "manifest.json").exists()


def test_missing_seed_residue_fails_closed(tmp_path: Path) -> None:
    pdb_dir = tmp_path / "pdb"
    pocket_dir = tmp_path / "pockets"
    pdb_dir.mkdir()
    pocket_dir.mkdir()
    _write_pdb(pdb_dir / "P12345_clean.pdb")
    _write_prediction(pocket_dir / "P12345_clean.pdb_predictions.csv", ["p1,1,7.5,A_99\n"])
    (tmp_path / "index.csv").write_text("stale\n")

    result = _run(tmp_path)

    assert result.returncode != 0
    assert "Declared seed residue A_99 is missing" in result.stderr
    assert not (tmp_path / "index.csv").exists()


def test_duplicate_rank_one_fails_closed(tmp_path: Path) -> None:
    pdb_dir = tmp_path / "pdb"
    pocket_dir = tmp_path / "pockets"
    pdb_dir.mkdir()
    pocket_dir.mkdir()
    _write_pdb(pdb_dir / "P12345_clean.pdb")
    _write_prediction(
        pocket_dir / "P12345_clean.pdb_predictions.csv",
        ["p1,1,7.5,A_8\n", "p2,1,6.5,A_9\n"],
    )

    result = _run(tmp_path)

    assert result.returncode != 0
    assert "exactly one rank=1 row" in result.stderr
    assert not (tmp_path / "index.csv").exists()


def test_header_only_prediction_is_audited_no_pocket_exclusion(tmp_path: Path) -> None:
    pdb_dir = tmp_path / "pdb"
    pocket_dir = tmp_path / "pockets"
    pdb_dir.mkdir()
    pocket_dir.mkdir()
    _write_pdb(pdb_dir / "P12345_clean.pdb")
    _write_prediction(pocket_dir / "P12345_clean.pdb_predictions.csv", [])

    result = _run(tmp_path)

    assert result.returncode == 0, result.stderr
    assert pd.read_csv(tmp_path / "index.csv").empty
    assert pd.read_csv(tmp_path / "exclusions.csv").to_dict("records") == [
        {"uniprot": "P12345", "reason": "no_ranked_pocket"}
    ]


def test_target_manifest_provenance_mismatch_fails_and_cleans_stale(tmp_path: Path) -> None:
    pdb_dir = tmp_path / "pdb"
    pocket_dir = tmp_path / "pockets"
    pdb_dir.mkdir()
    pocket_dir.mkdir()
    _write_pdb(pdb_dir / "P12345_clean.pdb")
    _write_prediction(pocket_dir / "P12345_clean.pdb_predictions.csv")
    targets_csv, manifest = _write_targets_and_manifest(tmp_path, ["P12345"])
    payload = json.loads(manifest.read_text())
    payload["artifact"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload) + "\n")
    params = tmp_path / "p2rank.params"
    params.write_text("x=1\n")
    (tmp_path / "index.csv").write_text("stale\n")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--targets-csv",
            str(targets_csv),
            "--target-cluster-manifest",
            str(manifest),
            "--pdb-dir",
            str(pdb_dir),
            "--pocket-dir",
            str(pocket_dir),
            "--p2rank-params",
            str(params),
            "--p2rank-version",
            "2.5",
            "--out-dir",
            str(tmp_path / "fragments_out"),
            "--out-index",
            str(tmp_path / "index.csv"),
            "--out-exclusions",
            str(tmp_path / "exclusions.csv"),
            "--out-manifest",
            str(tmp_path / "manifest.json"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "artifact sha256 does not match" in result.stderr
    assert not (tmp_path / "index.csv").exists()
