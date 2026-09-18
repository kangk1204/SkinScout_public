"""Regression tests for Stage 0 P2Rank batch/postprocess gates."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run_postprocess(tmp_path: Path, extra: list[str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage0_p2rank_postprocess.py"),
            "--pocket-dir", str(tmp_path / "pockets"),
            "--no-pocket-out", str(tmp_path / "no_pocket.list"),
            *(extra or []),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def write_prediction(path: Path, score: float) -> None:
    path.write_text(
        "name,rank,score,druggability_score,center_x,center_y,center_z,radius\n"
        f"p1,1,{score},{score},1.0,2.0,3.0,10.0\n"
    )


def test_p2rank_batch_fails_when_no_cleaned_pdb_inputs(tmp_path: Path) -> None:
    clean = tmp_path / "clean"
    pockets = tmp_path / "pockets"
    clean.mkdir()

    res = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/stage0_p2rank_batch.sh"),
            str(clean),
            str(pockets),
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "no cleaned PDB inputs found" in res.stderr


def test_p2rank_batch_removes_stale_outputs_before_run(tmp_path: Path) -> None:
    clean = tmp_path / "clean"
    pockets = tmp_path / "pockets"
    fake_bin = tmp_path / "bin"
    clean.mkdir()
    pockets.mkdir()
    fake_bin.mkdir()
    (clean / "P12345_clean.pdb").write_text("ATOM\n")
    (pockets / "STALE_clean.pdb_predictions.csv").write_text("stale\n")
    (pockets / "STALE.pockets.json").write_text('{"stale": true}\n')
    nested = pockets / "old_dataset"
    nested.mkdir()
    (nested / "NESTED_clean.pdb_predictions.csv").write_text("stale\n")
    (nested / "NESTED.pockets.json").write_text('{"stale": true}\n')
    fake_prank = fake_bin / "prank"
    fake_prank.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "ds=\"$2\"\n"
        "out=\"\"\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  case \"$1\" in\n"
        "    -o) out=\"$2\"; shift 2 ;;\n"
        "    *) shift ;;\n"
        "  esac\n"
        "done\n"
        "while IFS= read -r pdb; do\n"
        "  base=\"$(basename \"$pdb\")\"\n"
        "  uid=\"${base%_clean.pdb}\"\n"
        "  printf 'name,rank,score,druggability_score,center_x,center_y,center_z,radius\\n' > \"$out/${uid}_clean.pdb_predictions.csv\"\n"
        "  printf 'p1,1,5.0,5.0,1.0,2.0,3.0,10.0\\n' >> \"$out/${uid}_clean.pdb_predictions.csv\"\n"
        "done < \"$ds\"\n"
    )
    fake_prank.chmod(0o755)

    res = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/stage0_p2rank_batch.sh"),
            str(clean),
            str(pockets),
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}"},
    )

    assert res.returncode == 0, res.stderr
    assert not (pockets / "STALE_clean.pdb_predictions.csv").exists()
    assert not (pockets / "STALE.pockets.json").exists()
    assert not (nested / "NESTED_clean.pdb_predictions.csv").exists()
    assert not (nested / "NESTED.pockets.json").exists()
    assert (pockets / "P12345_clean.pdb_predictions.csv").exists()


def test_p2rank_postprocess_fails_without_prediction_csvs(tmp_path: Path) -> None:
    (tmp_path / "pockets").mkdir()
    (tmp_path / "no_pocket.list").write_text("STALE\n")

    res = run_postprocess(tmp_path)

    assert res.returncode != 0
    assert "No P2Rank prediction CSV files were processed" in res.stderr
    assert not (tmp_path / "no_pocket.list").exists()


def test_p2rank_postprocess_fails_when_processed_count_too_low(tmp_path: Path) -> None:
    pockets = tmp_path / "pockets"
    pockets.mkdir()
    write_prediction(pockets / "P12345_clean.pdb_predictions.csv", 5.0)
    (pockets / "P12345.pockets.json").write_text('{"stale": true}')
    (tmp_path / "no_pocket.list").write_text("STALE\n")

    res = run_postprocess(tmp_path, ["--min-processed-count", "2"])

    assert res.returncode != 0
    assert "1 processed; required at least 2" in res.stderr
    assert not (tmp_path / "no_pocket.list").exists()
    assert not (pockets / "P12345.pockets.json").exists()


def test_p2rank_postprocess_fails_when_usable_fraction_too_low(tmp_path: Path) -> None:
    pockets = tmp_path / "pockets"
    pockets.mkdir()
    write_prediction(pockets / "P12345_clean.pdb_predictions.csv", 0.1)
    write_prediction(pockets / "P67890_clean.pdb_predictions.csv", 5.0)
    (pockets / "P12345.pockets.json").write_text('{"stale": true}')
    (pockets / "P67890.pockets.json").write_text('{"stale": true}')
    (tmp_path / "no_pocket.list").write_text("STALE\n")

    res = run_postprocess(tmp_path, ["--min-usable-fraction", "0.75"])

    assert res.returncode != 0
    assert "1/2 with usable pockets" in res.stderr
    assert not (tmp_path / "no_pocket.list").exists()
    assert not (pockets / "P12345.pockets.json").exists()
    assert not (pockets / "P67890.pockets.json").exists()


def test_p2rank_postprocess_rejects_boolean_pocket_score(tmp_path: Path) -> None:
    pockets = tmp_path / "pockets"
    pockets.mkdir()
    (pockets / "P12345_clean.pdb_predictions.csv").write_text(
        "name,rank,score,druggability_score,center_x,center_y,center_z,radius\n"
        "p1,1,5.0,0.8,1.0,2.0,3.0,10.0\n"
        "p2,2,True,0.7,4.0,5.0,6.0,9.0\n"
    )
    (pockets / "P12345.pockets.json").write_text('{"stale": true}')
    (tmp_path / "no_pocket.list").write_text("STALE\n")

    res = run_postprocess(tmp_path)

    assert res.returncode != 0
    assert (
        "Invalid P2Rank prediction row 2 in P12345_clean.pdb_predictions.csv: "
        "score must be numeric"
    ) in res.stderr
    assert not (tmp_path / "no_pocket.list").exists()
    assert not (pockets / "P12345.pockets.json").exists()


def test_p2rank_postprocess_fails_when_expected_input_is_missing(tmp_path: Path) -> None:
    pockets = tmp_path / "pockets"
    pockets.mkdir()
    write_prediction(pockets / "P12345_clean.pdb_predictions.csv", 5.0)
    input_list = pockets / "proteins.ds"
    input_list.write_text(
        f"{tmp_path / 'clean' / 'P12345_clean.pdb'}\n"
        f"{tmp_path / 'clean' / 'P67890_clean.pdb'}\n"
    )
    (tmp_path / "no_pocket.list").write_text("STALE\n")

    res = run_postprocess(
        tmp_path,
        ["--expected-input-list", str(input_list), "--min-processed-count", "1"],
    )

    assert res.returncode != 0
    assert "1 missing, 0 unexpected" in res.stderr
    assert not (tmp_path / "no_pocket.list").exists()
    assert not (pockets / "P12345.pockets.json").exists()


def test_p2rank_postprocess_rejects_duplicate_uid_prediction_csvs(
    tmp_path: Path,
) -> None:
    pockets = tmp_path / "pockets"
    nested = pockets / "stale_dataset"
    nested.mkdir(parents=True)
    write_prediction(pockets / "P12345_clean.pdb_predictions.csv", 5.0)
    write_prediction(nested / "P12345_clean.pdb_predictions.csv", 0.1)
    (pockets / "P12345.pockets.json").write_text('{"stale": true}')
    (tmp_path / "no_pocket.list").write_text("STALE\n")

    res = run_postprocess(tmp_path)

    assert res.returncode != 0
    assert "Duplicate P2Rank prediction CSVs" in res.stderr
    assert not (tmp_path / "no_pocket.list").exists()
    assert not (pockets / "P12345.pockets.json").exists()


def test_p2rank_postprocess_writes_manifests_when_quality_gates_pass(tmp_path: Path) -> None:
    pockets = tmp_path / "pockets"
    pockets.mkdir()
    write_prediction(pockets / "P12345_clean.pdb_predictions.csv", 5.0)
    write_prediction(pockets / "P67890_clean.pdb_predictions.csv", 0.1)

    res = run_postprocess(
        tmp_path,
        ["--min-processed-count", "2", "--min-usable-fraction", "0.5"],
    )

    assert res.returncode == 0, res.stderr
    assert (pockets / "P12345.pockets.json").exists()
    assert (pockets / "P67890.pockets.json").exists()
    assert (tmp_path / "no_pocket.list").read_text() == "P67890\n"
