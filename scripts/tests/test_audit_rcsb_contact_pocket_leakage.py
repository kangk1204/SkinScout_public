"""Tests for the coordinate-level RCSB contact-pocket leakage auditor."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "eval" / "audit_rcsb_contact_pocket_leakage.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _fragment(path: Path, chain: str = "A") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"ATOM      1  N   ALA {chain}   1       0.000   0.000   0.000  1.00 20.00           N",
                f"ATOM      2  CA  ALA {chain}   1       1.000   0.000   0.000  1.00 20.00           C",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _mock_foldseek(tmp_path: Path) -> Path:
    binary = tmp_path / "mock_foldseek.py"
    binary.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

record = Path(os.environ["MOCK_FOLDSEEK_RECORD"])
mode = os.environ.get("MOCK_FOLDSEEK_MODE", "ok")
if len(sys.argv) > 1 and sys.argv[1] == "version":
    with record.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"argv": sys.argv[1:]}) + "\\n")
    print("foldseek mock 1.0")
    raise SystemExit(0)
if len(sys.argv) <= 1 or sys.argv[1] != "easy-search":
    print("unexpected command", file=sys.stderr)
    raise SystemExit(2)
if mode == "nonzero":
    print("forced failure", file=sys.stderr)
    raise SystemExit(7)
query_dir = Path(sys.argv[2])
target_dir = Path(sys.argv[3])
out = Path(sys.argv[4])
queries = sorted(path.stem for path in query_dir.glob("*.pdb"))
targets = sorted(path.stem for path in target_dir.glob("*.pdb"))
with record.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({"argv": sys.argv[1:], "query_files": queries, "target_files": targets}) + "\\n")
if mode == "malformed":
    out.write_text("too\\tfew\\n", encoding="utf-8")
    raise SystemExit(0)
if mode == "nan":
    out.write_text(f"{queries[0]}_A\\t{targets[0]}_B\\tnan\\t0.5\\t0.5\\t1e-4\\t20\\t10\\t10\\t10\\n", encoding="utf-8")
    raise SystemExit(0)
if mode == "aln_gt1":
    out.write_text(f"{queries[0]}_A\\t{targets[0]}_B\\t1.15\\t0.7\\t0.45\\t1e-4\\t20\\t10\\t10\\t10\\n", encoding="utf-8")
    raise SystemExit(0)
if mode == "qtm_gt1":
    out.write_text(f"{queries[0]}_A\\t{targets[0]}_B\\t0.6\\t1.01\\t0.45\\t1e-4\\t20\\t10\\t10\\t10\\n", encoding="utf-8")
    raise SystemExit(0)
if mode == "unknown_id":
    out.write_text(f"{queries[0]}_A\\tmissing_A\\t0.5\\t0.5\\t0.5\\t1e-4\\t20\\t10\\t10\\t10\\n", encoding="utf-8")
    raise SystemExit(0)
rows = []
if queries and targets:
    rows.append(f"{queries[0]}_A\\t{targets[0]}_B\\t0.51\\t0.70\\t0.45\\t1e-8\\t90.5\\t42\\t50\\t60")
out.write_text("\\n".join(rows) + ("\\n" if rows else ""), encoding="utf-8")
""",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    return binary


def _write_benchmark(tmp_path: Path) -> tuple[Path, Path, Path]:
    train = pd.DataFrame(
        [
            {"evidence_id": "ev-train-1", "split": "train", "uniprot": "P11111", "source_db": "ChEMBL"},
            {"evidence_id": "ev-train-2", "split": "train", "uniprot": "P22222", "source_db": "BindingDB"},
        ]
    )
    dev = pd.DataFrame(
        [{"evidence_id": "ev-dev-1", "split": "dev", "uniprot": "P33333", "source_db": "GtoPdb"}]
    )
    train_path = tmp_path / "train.parquet"
    dev_path = tmp_path / "dev.parquet"
    train.to_parquet(train_path, index=False)
    dev.to_parquet(dev_path, index=False)
    manifest = tmp_path / "activity_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "activity_benchmark.v1",
                "output_sha256": {
                    "train.parquet": _sha256(train_path),
                    "dev.parquet": _sha256(dev_path),
                },
                "splits": {"counts": {"train": len(train), "dev": len(dev)}},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest, train_path, dev_path


def _write_rcsb(tmp_path: Path) -> tuple[Path, Path, Path]:
    fragment_dir = tmp_path / "rcsb_fragments"
    rows = []
    for pair_id, entry, ligand in (("pair-1", "1ABC", "lig-a"), ("pair-2", "2DEF", "lig-b")):
        rel = f"{pair_id}.pdb"
        path = fragment_dir / rel
        _fragment(path)
        rows.append(
            {
                "pair_id": pair_id,
                "entry_id": entry,
                "component_id": "LIG",
                "instance_id": f"{entry}.A",
                "uniprot": "Q99999",
                "ligand_key": ligand,
                "fragment_path": rel,
                "fragment_sha256": _sha256(path),
            }
        )
    index = tmp_path / "rcsb_index.csv"
    _write_csv(
        index,
        rows,
        ["pair_id", "entry_id", "component_id", "instance_id", "uniprot", "ligand_key", "fragment_path", "fragment_sha256"],
    )
    exclusions = tmp_path / "rcsb_exclusions.csv"
    exclusion_rows = [
        {
            "pair_id": "pair-3",
            "entry_id": "3GHI",
            "component_id": "LIG",
            "instance_id": "3GHI.A",
            "uniprot": "Q88888",
            "ligand_key": "lig-c",
            "reason": "contact_residues_not_extractable",
            "detail": "synthetic",
        }
    ]
    _write_csv(
        exclusions,
        exclusion_rows,
        [
            "pair_id",
            "entry_id",
            "component_id",
            "instance_id",
            "uniprot",
            "ligand_key",
            "reason",
            "detail",
        ],
    )
    manifest = tmp_path / "rcsb_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "skinscout.rcsb-contact-pocket-fragments.v1",
                "contract": {
                    "never_training": True,
                    "never_calibration": True,
                    "never_model_selection": True,
                    "strict_pairs_only": True,
                },
                "counts": {
                    "strict_dual_cold_pairs": len(rows) + len(exclusion_rows),
                    "indexed": len(rows),
                    "excluded": len(exclusion_rows),
                    "unique_entries": 3,
                },
                "artifacts": {
                    "index_csv": {"path": str(index.resolve()), "sha256": _sha256(index), "rows": len(rows)},
                    "fragment_dir": {
                        "path": str(fragment_dir.resolve()),
                        "tree_sha256": _tree_digest(fragment_dir),
                        "files": len(rows),
                    },
                    "exclusions_csv": {
                        "path": str(exclusions.resolve()),
                        "sha256": _sha256(exclusions),
                        "rows": len(exclusion_rows),
                    },
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return index, manifest, fragment_dir


def _write_prior(tmp_path: Path) -> tuple[Path, Path, Path]:
    fragment_dir = tmp_path / "prior_fragments"
    rows = []
    for uniprot in ("P11111", "P33333", "P99999"):
        rel = f"fragments/{uniprot}_p2rank_top1_context2.pdb"
        path = fragment_dir / rel
        _fragment(path)
        rows.append({"uniprot": uniprot, "fragment_path": rel, "fragment_sha256": _sha256(path)})
    index = tmp_path / "prior_index.csv"
    _write_csv(index, rows, ["uniprot", "fragment_path", "fragment_sha256"])
    manifest = tmp_path / "prior_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "skinscout.pocket-fragment-universe.v1",
                "counts": {"accepted": len(rows)},
                "artifacts": {
                    "index": {"path": str(index.resolve()), "sha256": _sha256(index), "rows": len(rows)},
                    "out_dir": str(fragment_dir.resolve()),
                    "out_dir_tree_sha256": _tree_digest(fragment_dir),
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return index, manifest, fragment_dir


def _fixture(tmp_path: Path) -> dict[str, Path]:
    activity_manifest, train, dev = _write_benchmark(tmp_path)
    rcsb_index, rcsb_manifest, rcsb_fragments = _write_rcsb(tmp_path)
    prior_index, prior_manifest, prior_fragments = _write_prior(tmp_path)
    return {
        "activity_manifest": activity_manifest,
        "train": train,
        "dev": dev,
        "rcsb_index": rcsb_index,
        "rcsb_manifest": rcsb_manifest,
        "rcsb_fragments": rcsb_fragments,
        "prior_index": prior_index,
        "prior_manifest": prior_manifest,
        "prior_fragments": prior_fragments,
    }


def _run(tmp_path: Path, paths: dict[str, Path], mode: str = "ok") -> subprocess.CompletedProcess[str]:
    record = tmp_path / "foldseek_argv.jsonl"
    env = {**os.environ, "MOCK_FOLDSEEK_RECORD": str(record), "MOCK_FOLDSEEK_MODE": mode}
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--foldseek",
            str(_mock_foldseek(tmp_path)),
            "--threads",
            "3",
            "--activity-benchmark-manifest",
            str(paths["activity_manifest"]),
            "--train-parquet",
            str(paths["train"]),
            "--dev-parquet",
            str(paths["dev"]),
            "--rcsb-index",
            str(paths["rcsb_index"]),
            "--rcsb-manifest",
            str(paths["rcsb_manifest"]),
            "--rcsb-fragment-dir",
            str(paths["rcsb_fragments"]),
            "--prior-index",
            str(paths["prior_index"]),
            "--prior-manifest",
            str(paths["prior_manifest"]),
            "--prior-fragment-dir",
            str(paths["prior_fragments"]),
            "--out-csv",
            str(tmp_path / "audit.csv"),
            "--out-manifest",
            str(tmp_path / "audit.manifest.json"),
        ],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def test_provenance_tamper_fails_closed_and_removes_outputs(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    (paths["prior_fragments"] / "fragments/P11111_p2rank_top1_context2.pdb").write_text("tampered\n", encoding="utf-8")
    (tmp_path / "audit.csv").write_text("stale\n", encoding="utf-8")
    (tmp_path / "audit.manifest.json").write_text("stale\n", encoding="utf-8")

    result = _run(tmp_path, paths)

    assert result.returncode == 1
    assert "fragment tree hash does not match" in result.stderr
    assert not (tmp_path / "audit.csv").exists()
    assert not (tmp_path / "audit.manifest.json").exists()


def test_snakemake_directory_markers_do_not_change_bound_tree_digest(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    (paths["rcsb_fragments"] / ".snakemake_timestamp").touch()
    (paths["prior_fragments"] / ".snakemake_timestamp").touch()

    result = _run(tmp_path, paths)

    assert result.returncode == 0, result.stderr


def test_command_options_chain_identifier_parsing_and_filtered_prior_universe(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    result = _run(tmp_path, paths)

    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in (tmp_path / "foldseek_argv.jsonl").read_text().splitlines()]
    assert calls[0]["argv"] == ["version"]
    command = calls[1]["argv"]
    assert command[0] == "easy-search"
    assert command[5:] == [
        "--threads",
        "3",
        "--alignment-type",
        "1",
        "--exhaustive-search",
        "1",
        "--exact-tmscore",
        "1",
        "--tmalign-hit-order",
        "4",
        "--format-output",
        "query,target,alntmscore,qtmscore,ttmscore,evalue,bits,alnlen,qlen,tlen",
    ]
    target_files = calls[1]["target_files"]
    assert len(target_files) == 2
    assert all("P99999" not in name for name in target_files)
    manifest = json.loads((tmp_path / "audit.manifest.json").read_text())
    assert manifest["prior_universe"]["filtered_prior_index_rows"] == 2
    assert manifest["prior_universe"]["missing_train_dev_uniprots_from_prior_index"] == ["P22222"]


def test_exact_metrics_coverage_leakage_and_contract_manifest(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    result = _run(tmp_path, paths)

    assert result.returncode == 0, result.stderr
    rows = _read_csv(tmp_path / "audit.csv")
    assert len(rows) == 3
    first = rows[0]
    second = next(row for row in rows if row["pair_id"] == "pair-2")
    excluded = next(row for row in rows if row["pair_id"] == "pair-3")
    assert first["pair_id"] == "pair-1"
    assert first["foldseek_covered"] == "true"
    assert first["best_prior_uniprot"] == "P11111"
    assert first["best_alntmscore"] == "0.51"
    assert first["best_qtmscore"] == "0.7"
    assert first["best_ttmscore"] == "0.45"
    assert first["best_max_directional_tm"] == "0.7"
    assert first["best_min_directional_tm"] == "0.45"
    assert first["leakage_tm40"] == "true"
    assert first["leakage_tm50"] == "true"
    assert first["leakage_tm60"] == "true"
    assert second["foldseek_covered"] == "false"
    assert second["best_prior_uniprot"] == ""
    assert second["leakage_tm40"] == "false"
    assert excluded["fragment_extracted"] == "false"
    assert excluded["foldseek_covered"] == "false"
    assert excluded["uncovered_reason"] == (
        "fragment_extraction:contact_residues_not_extractable"
    )

    manifest = json.loads((tmp_path / "audit.manifest.json").read_text())
    assert manifest["contract"]["evaluation_only"] is True
    assert manifest["contract"]["never_training"] is True
    assert manifest["contract"]["never_calibration"] is True
    assert manifest["contract"]["never_model_selection"] is True
    assert manifest["summary"]["pairs"]["total"] == 3
    assert manifest["summary"]["pairs"]["fragment_extracted"] == 2
    assert manifest["summary"]["pairs"]["fragment_extraction_excluded"] == 1
    assert manifest["summary"]["pairs"]["covered"] == 1
    assert manifest["summary"]["pairs"]["uncovered"] == 2
    assert manifest["summary"]["pairs"]["leakage_by_max_directional_tm_threshold"]["0.6"]["count"] == 1
    assert manifest["outputs"]["detailed_csv"]["sha256"] == _sha256(tmp_path / "audit.csv")


def test_malformed_foldseek_output_fails_and_cleans_outputs(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    (tmp_path / "audit.csv").write_text("stale\n", encoding="utf-8")

    result = _run(tmp_path, paths, mode="malformed")

    assert result.returncode == 1
    assert "expected 10" in result.stderr
    assert not (tmp_path / "audit.csv").exists()
    assert not (tmp_path / "audit.manifest.json").exists()

    nonfinite = _run(tmp_path, paths, mode="nan")
    assert nonfinite.returncode == 1
    assert "non-finite alntmscore" in nonfinite.stderr
    assert not (tmp_path / "audit.csv").exists()


def test_alignment_normalized_tm_above_one_is_diagnostic_only(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    result = _run(tmp_path, paths, mode="aln_gt1")

    assert result.returncode == 0, result.stderr
    first = _read_csv(tmp_path / "audit.csv")[0]
    assert first["best_alntmscore"] == "1.15"
    assert first["best_max_directional_tm"] == "0.7"


def test_directional_tm_above_one_fails_closed(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    result = _run(tmp_path, paths, mode="qtm_gt1")

    assert result.returncode == 1
    assert "directional TM-score outside [0, 1]" in result.stderr
    assert not (tmp_path / "audit.csv").exists()


def test_nonzero_foldseek_and_unknown_identifier_fail_closed(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    nonzero = _run(tmp_path, paths, mode="nonzero")
    assert nonzero.returncode == 1
    assert "Foldseek easy-search failed" in nonzero.stderr
    assert not (tmp_path / "audit.csv").exists()

    unknown = _run(tmp_path, paths, mode="unknown_id")
    assert unknown.returncode == 1
    assert "identifier is unknown" in unknown.stderr
    assert not (tmp_path / "audit.csv").exists()
