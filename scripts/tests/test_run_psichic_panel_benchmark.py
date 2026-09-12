"""Regression tests for the direct PSICHIC known-target panel audit."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "eval" / "run_psichic_panel_benchmark.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_panel(path: Path, rows: list[dict[str, str]]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_psichic_case(
    root: Path,
    case_id: str,
    rows: list[dict[str, object]],
    *,
    metadata: dict[str, object] | None = None,
) -> Path:
    case_dir = root / case_id
    case_dir.mkdir(parents=True)
    tsv = case_dir / "psichic_proteome.tsv"
    pd.DataFrame(rows).to_csv(tsv, sep="\t", index=False)
    if metadata is not None:
        (case_dir / "psichic_proteome.metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
    return tsv


def run_benchmark(
    *,
    panel: Path,
    psichic_root: Path,
    out_dir: Path,
    source_template: str | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
            sys.executable,
            str(SCRIPT),
            "--panel-csv",
            str(panel),
            "--psichic-root",
            str(psichic_root),
            "--out-dir",
            str(out_dir),
        ]
    if source_template is not None:
        command.extend(["--source-template", source_template])
    return subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_missing_ranking_is_not_counted_as_missing_source_or_no_case_coverage(
    tmp_path: Path,
) -> None:
    panel = write_panel(
        tmp_path / "panel.csv",
        [{"case_id": "case_a", "known_targets": "P99999"}],
    )
    psichic_root = tmp_path / "psichic"
    write_psichic_case(
        psichic_root,
        "case_a",
        [{"target_id": "P11111", "psichic_score": 0.9}],
    )
    out_dir = tmp_path / "audit"

    res = run_benchmark(panel=panel, psichic_root=psichic_root, out_dir=out_dir)

    assert res.returncode == 0, res.stderr
    cases = pd.read_csv(out_dir / "psichic_direct_panel_cases.csv")
    assert cases.loc[0, "source_status"] == "known_target_missing_from_ranking"
    assert bool(cases.loc[0, "covered_by_source"]) is True
    assert bool(cases.loc[0, "known_target_ranked"]) is False
    summary = json.loads((out_dir / "psichic_direct_panel_summary.json").read_text())
    assert summary["panel_denominator_cases"] == 1
    assert summary["source_covered_cases"] == 1
    assert summary["missing_source_cases"] == 0
    assert summary["no_case_coverage_cases"] == 0
    assert summary["known_target_missing_from_covered_ranking_cases"] == 1
    assert summary["case_top10_fraction_of_full_panel"] == 0
    assert summary["target_pair_top30_fraction"] == 0


def test_conversion_sorts_descending_score_with_uniprot_tie_break(tmp_path: Path) -> None:
    panel = write_panel(
        tmp_path / "panel.csv",
        [{"case_id": "case_a", "known_targets": "P22222"}],
    )
    psichic_root = tmp_path / "psichic"
    write_psichic_case(
        psichic_root,
        "case_a",
        [
            {"target_id": "P33333", "psichic_score": 0.2},
            {"target_id": "P22222", "psichic_score": 0.8},
            {"target_id": "P11111", "psichic_score": 0.8},
        ],
    )
    out_dir = tmp_path / "audit"

    res = run_benchmark(panel=panel, psichic_root=psichic_root, out_dir=out_dir)

    assert res.returncode == 0, res.stderr
    ranking = pd.read_csv(
        out_dir / "rankings" / "case_a__psichic_direct_ranked_targets.csv"
    )
    assert list(ranking["target_id"]) == ["P11111", "P22222", "P33333"]
    assert list(ranking["rank"]) == [1, 2, 3]
    cases = pd.read_csv(out_dir / "psichic_direct_panel_cases.csv")
    assert cases.loc[0, "best_known_target_rank"] == 2
    assert cases.loc[0, "best_known_target_score"] == 0.8
    targets = pd.read_csv(out_dir / "psichic_direct_panel_targets.csv")
    assert targets.loc[0, "target_rank"] == 2
    assert bool(targets.loc[0, "top10"]) is True


def test_coverage_denominator_preserves_full_panel_and_missing_source(
    tmp_path: Path,
) -> None:
    panel = write_panel(
        tmp_path / "panel.csv",
        [
            {"case_id": "covered", "known_targets": "P11111"},
            {"case_id": "empty", "known_targets": "P22222"},
            {"case_id": "missing", "known_targets": "P33333"},
        ],
    )
    psichic_root = tmp_path / "psichic"
    write_psichic_case(
        psichic_root,
        "covered",
        [{"target_id": "P11111", "psichic_score": 0.7}],
    )
    write_psichic_case(psichic_root, "empty", [])
    out_dir = tmp_path / "audit"

    res = run_benchmark(panel=panel, psichic_root=psichic_root, out_dir=out_dir)

    assert res.returncode == 0, res.stderr
    summary = json.loads((out_dir / "psichic_direct_panel_summary.json").read_text())
    assert summary["panel_denominator_cases"] == 3
    assert summary["source_tsv_present_cases"] == 2
    assert summary["source_covered_cases"] == 1
    assert summary["known_target_ranked_cases"] == 1
    assert summary["missing_source_cases"] == 1
    assert summary["no_case_coverage_cases"] == 1
    assert summary["coverage_fraction_of_full_panel"] == 1 / 3
    assert summary["known_target_ranked_fraction_of_full_panel"] == 1 / 3
    assert summary["target_pair_denominator"] == 3


def test_known_target_rank_records_best_target_rank(tmp_path: Path) -> None:
    panel = write_panel(
        tmp_path / "panel.csv",
        [{"case_id": "case_a", "known_targets": "P22222;P33333"}],
    )
    psichic_root = tmp_path / "psichic"
    write_psichic_case(
        psichic_root,
        "case_a",
        [
            {"target_id": "P11111", "psichic_score": 0.95},
            {"target_id": "P22222", "psichic_score": 0.7},
            {"target_id": "P33333", "psichic_score": 0.9},
        ],
    )
    out_dir = tmp_path / "audit"

    res = run_benchmark(panel=panel, psichic_root=psichic_root, out_dir=out_dir)

    assert res.returncode == 0, res.stderr
    cases = pd.read_csv(out_dir / "psichic_direct_panel_cases.csv")
    assert cases.loc[0, "source_status"] == "known_target_ranked"
    assert bool(cases.loc[0, "known_target_ranked"]) is True
    assert cases.loc[0, "best_known_target"] == "P33333"
    assert cases.loc[0, "best_known_target_rank"] == 2
    targets = pd.read_csv(out_dir / "psichic_direct_panel_targets.csv")
    assert list(targets["target_rank"]) == [3, 2]
    assert summary_metrics(out_dir)["target_pair_mrr"] == (1 / 3 + 1 / 2) / 2


def test_manifest_records_hash_provenance_model_and_non_probability_semantics(
    tmp_path: Path,
) -> None:
    panel = write_panel(
        tmp_path / "panel.csv",
        [{"case_id": "case_a", "known_targets": "P11111"}],
    )
    psichic_root = tmp_path / "psichic"
    tsv = write_psichic_case(
        psichic_root,
        "case_a",
        [{"target_id": "P11111", "psichic_score": 0.7}],
        metadata={"model": "PDBv2020_PSICHIC", "checkpoint": "fixture.ckpt"},
    )
    out_dir = tmp_path / "audit"

    res = run_benchmark(panel=panel, psichic_root=psichic_root, out_dir=out_dir)

    assert res.returncode == 0, res.stderr
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["schema_version"] == "skinscout.psichic-direct-panel-audit.v1"
    assert manifest["score_semantics"].endswith("not_calibrated_probability")
    assert "probability" in manifest["score_semantics"]
    assert "calibrated_probability" not in manifest["score_semantics"].replace(
        "not_calibrated_probability", ""
    )
    source = manifest["sources"][0]
    assert source["sha256"] == sha256(tsv)
    assert source["model_metadata"]["model"] == "PDBv2020_PSICHIC"
    assert source["model_metadata"]["checkpoint"] == "fixture.ckpt"
    assert manifest["outputs"]["case_audit_csv_sha256"] == sha256(
        out_dir / "psichic_direct_panel_cases.csv"
    )
    assert manifest["outputs"]["target_audit_csv_sha256"] == sha256(
        out_dir / "psichic_direct_panel_targets.csv"
    )


def test_malformed_score_fails_closed_without_publishing_output(tmp_path: Path) -> None:
    panel = write_panel(
        tmp_path / "panel.csv",
        [{"case_id": "case_a", "known_targets": "P11111"}],
    )
    psichic_root = tmp_path / "psichic"
    write_psichic_case(
        psichic_root,
        "case_a",
        [{"target_id": "P11111", "psichic_score": "not-a-number"}],
    )
    out_dir = tmp_path / "audit"

    res = run_benchmark(panel=panel, psichic_root=psichic_root, out_dir=out_dir)

    assert res.returncode != 0
    assert "psichic_score column contains non-finite values" in res.stderr
    assert not out_dir.exists()
    assert not (tmp_path / ".audit.psichic-panel-staging").exists()


def test_nested_skin_scout_run_template_is_supported(tmp_path: Path) -> None:
    panel = write_panel(
        tmp_path / "panel.csv",
        [{"case_id": "case_a", "known_targets": "P11111"}],
    )
    psichic_root = tmp_path / "runs"
    nested = psichic_root / "deepdiag_case_a" / "03_targets" / "mode_fast"
    nested.mkdir(parents=True)
    tsv = nested / "psichic_proteome.tsv"
    pd.DataFrame(
        [{"target_id": "P11111", "psichic_score": 0.7}]
    ).to_csv(tsv, sep="\t", index=False)
    out_dir = tmp_path / "audit"

    res = run_benchmark(
        panel=panel,
        psichic_root=psichic_root,
        out_dir=out_dir,
        source_template="deepdiag_{case_id}/03_targets/mode_fast/psichic_proteome.tsv",
    )

    assert res.returncode == 0, res.stderr
    cases = pd.read_csv(out_dir / "psichic_direct_panel_cases.csv")
    assert cases.loc[0, "best_known_target_rank"] == 1
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["inputs"]["source_template"].startswith("deepdiag_{case_id}")


def test_blank_panel_values_and_unsafe_template_fail_closed(tmp_path: Path) -> None:
    panel = write_panel(
        tmp_path / "panel.csv",
        [{"case_id": "case_a", "known_targets": ""}],
    )
    psichic_root = tmp_path / "runs"
    psichic_root.mkdir()

    blank = run_benchmark(
        panel=panel, psichic_root=psichic_root, out_dir=tmp_path / "blank"
    )
    assert blank.returncode != 0
    assert "known_targets is blank" in blank.stderr

    valid_panel = write_panel(
        tmp_path / "valid.csv",
        [{"case_id": "case_a", "known_targets": "P11111"}],
    )
    unsafe = run_benchmark(
        panel=valid_panel,
        psichic_root=psichic_root,
        out_dir=tmp_path / "unsafe",
        source_template="../{case_id}/psichic_proteome.tsv",
    )
    assert unsafe.returncode != 0
    assert "relative path without '..'" in unsafe.stderr


def summary_metrics(out_dir: Path) -> dict[str, object]:
    return json.loads((out_dir / "psichic_direct_panel_summary.json").read_text())
