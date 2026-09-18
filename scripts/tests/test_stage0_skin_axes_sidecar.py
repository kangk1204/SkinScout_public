"""Regression tests for the SkinScore axes sidecar completion contract (C13).

The producer writes ``skin_score.tsv`` plus ``skin_score.tsv.axes.json``. Stage 0
completion must require the sidecar and validate score semantics, normalisation,
declared/effective weights, and consistency when an axis is missing.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from stage0_verify import chk_skin_score_axes  # noqa: E402


def _write_hpa_sources(hpa: Path) -> None:
    hpa.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "Gene": ["KRT14", "TYR", "HOUSE1"],
        "Tissue": ["skin", "skin", "skin"],
        "nTPM": [100.0, 80.0, 1.0],
    }).to_csv(hpa / "rna_tissue_consensus.tsv", sep="\t", index=False)
    pd.DataFrame({
        "Gene": ["KRT14", "TYR", "HOUSE1"],
        "Cell type": ["keratinocytes", "melanocytes", "adipocytes"],
        "nCPM": [50.0, 60.0, 90.0],
    }).to_csv(hpa / "rna_single_cell_type.tsv", sep="\t", index=False)
    pd.DataFrame({
        "gene": ["KRT14", "TYR", "HOUSE1"],
        "uniprot": ["P02533", "P14679", "P00001"],
    }).to_csv(hpa / "gene_to_uniprot.tsv", sep="\t", index=False)


def _write_aux_sources(root: Path) -> tuple[Path, Path, Path]:
    proteome = root / "proteome"
    proteome.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "uniprot": ["P02533", "P14679", "P00001"],
        "lfq": [9.0, 8.0, 1.0],
    }).to_csv(proteome / "skin_proteome.tsv", sep="\t", index=False)

    gtex = root / "gtex"
    gtex.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "uniprot": ["P02533", "P14679", "P00001"],
        "tpm": [120.0, 90.0, 1.0],
    }).to_csv(gtex / "skin_tpm.tsv", sep="\t", index=False)

    sc = root / "sc"
    sc.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "uniprot": ["P02533", "P14679", "P00001"],
        "max_expression": [60.0, 70.0, 1.0],
    }).to_csv(sc / "skin_max_celltype.tsv", sep="\t", index=False)
    return proteome, gtex, sc


def build_skin_score(tmp_path: Path, *, complete_sources: bool = False) -> Path:
    hpa = tmp_path / "hpa"
    _write_hpa_sources(hpa)
    if complete_sources:
        proteome, gtex, sc = _write_aux_sources(tmp_path)
    else:
        proteome = tmp_path / "missing_proteome"
        gtex = tmp_path / "missing_gtex"
        sc = tmp_path / "missing_sc"
    out = tmp_path / "skin_score.tsv"
    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "stage0_skin_score.py"),
            "--hpa-dir", str(hpa),
            "--proteome-dir", str(proteome),
            "--gtex-dir", str(gtex),
            "--sc-dir", str(sc),
            "--out-tsv", str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr
    return out


def sidecar_path(tsv_path: Path) -> Path:
    return tsv_path.with_suffix(tsv_path.suffix + ".axes.json")


def read_sidecar(tsv_path: Path) -> dict[str, object]:
    return json.loads(sidecar_path(tsv_path).read_text())


def write_sidecar(tsv_path: Path, payload: dict[str, object]) -> None:
    sidecar_path(tsv_path).write_text(json.dumps(payload))


def test_producer_sidecar_passes_completion_check_with_a_missing_axis(
    tmp_path: Path,
) -> None:
    tsv = build_skin_score(tmp_path)

    check = chk_skin_score_axes(tsv, "v3_skin_score_axes")

    assert check.ok, check.detail
    payload = read_sidecar(tsv)
    assert payload["schema_version"] == "skinscout.skin-score-axes.v2"
    assert payload["axes_absent"] == ["gtex", "proteome", "sc"]
    assert set(payload["effective_weights"]) == {"hpa_cell", "hpa_tissue"}
    assert abs(sum(payload["effective_weights"].values()) - 1.0) < 1e-6


def test_producer_sidecar_records_declared_weights_when_all_axes_present(
    tmp_path: Path,
) -> None:
    tsv = build_skin_score(tmp_path, complete_sources=True)

    check = chk_skin_score_axes(tsv, "v3_skin_score_axes")

    assert check.ok, check.detail
    payload = read_sidecar(tsv)
    assert payload["axes_absent"] == []
    assert payload["effective_weights"] == payload["declared_weights"]


def test_missing_sidecar_fails_completion_check(tmp_path: Path) -> None:
    tsv = build_skin_score(tmp_path)
    sidecar_path(tsv).unlink()

    check = chk_skin_score_axes(tsv, "v3_skin_score_axes")

    assert not check.ok
    assert "missing or empty" in check.detail


def test_old_schema_sidecar_fails_completion_check(tmp_path: Path) -> None:
    tsv = build_skin_score(tmp_path)
    payload = read_sidecar(tsv)
    payload["schema_version"] = "skinscout.skin-score-axes.v1"
    write_sidecar(tsv, payload)

    check = chk_skin_score_axes(tsv, "v3_skin_score_axes")

    assert not check.ok
    assert "schema_version" in check.detail


def test_nonfinite_declared_weight_fails_completion_check(tmp_path: Path) -> None:
    tsv = build_skin_score(tmp_path)
    payload = read_sidecar(tsv)
    payload["declared_weights"]["hpa_tissue"] = float("inf")
    write_sidecar(tsv, payload)

    check = chk_skin_score_axes(tsv, "v3_skin_score_axes")

    assert not check.ok
    assert "declared_weights" in check.detail


def test_nonfinite_effective_weight_fails_completion_check(tmp_path: Path) -> None:
    tsv = build_skin_score(tmp_path)
    payload = read_sidecar(tsv)
    first_axis = next(iter(payload["effective_weights"]))
    payload["effective_weights"][first_axis] = float("nan")
    write_sidecar(tsv, payload)

    check = chk_skin_score_axes(tsv, "v3_skin_score_axes")

    assert not check.ok
    assert "effective_weights.values" in check.detail


def test_effective_weights_must_match_axes_absent(tmp_path: Path) -> None:
    tsv = build_skin_score(tmp_path)
    payload = read_sidecar(tsv)
    payload["axes_absent"] = ["hpa_cell"]
    write_sidecar(tsv, payload)

    check = chk_skin_score_axes(tsv, "v3_skin_score_axes")

    assert not check.ok
    assert "effective_weights.keys" in check.detail


def test_effective_weights_must_renormalise_to_one(tmp_path: Path) -> None:
    tsv = build_skin_score(tmp_path)
    payload = read_sidecar(tsv)
    for axis in payload["effective_weights"]:
        payload["effective_weights"][axis] = 0.1
    write_sidecar(tsv, payload)

    check = chk_skin_score_axes(tsv, "v3_skin_score_axes")

    assert not check.ok
    assert "effective_weights.sum" in check.detail


def test_collect_checks_fails_completion_when_sidecar_is_stale(
    tmp_path: Path,
) -> None:
    from stage0_verify import collect_checks

    skin = tmp_path / "data" / "skin_expression"
    skin.mkdir(parents=True)
    (skin / "skin_score.tsv").write_text("uniprot\tskin_score\nP02533\t0.5\n")
    (skin / "skin_score.tsv.axes.json").write_text(
        json.dumps(
            {
                "schema_version": "skinscout.skin-score-axes.v1",
                "score_semantics": "relative_within_build_expression_context",
                "normalization": "per-axis min-max across proteins in this build",
                "declared_weights": {
                    "hpa_tissue": 0.30,
                    "hpa_cell": 0.25,
                    "proteome": 0.20,
                    "gtex": 0.10,
                    "sc": 0.15,
                },
                "effective_weights": {
                    "hpa_tissue": 0.30,
                    "hpa_cell": 0.25,
                    "proteome": 0.20,
                    "gtex": 0.10,
                    "sc": 0.15,
                },
                "axes_absent": [],
                "context_support_threshold": 0.2,
            }
        )
    )

    checks = collect_checks(tmp_path, claim_quality=False, check_stage0_flag=False)

    by_name = {check.name: check for check in checks}
    assert not by_name["v3_skin_score_axes"].ok
    assert "schema_version" in by_name["v3_skin_score_axes"].detail


def test_stage0_completion_rule_requires_the_sidecar() -> None:
    infra = (ROOT / "workflow" / "rules" / "stage0_infra.smk").read_text()
    stage0_flag = infra.split("rule stage0_complete_flag:", 1)[1]
    inputs = stage0_flag.split("output:", 1)[0]
    assert "skin_score.tsv.axes.json" in inputs

    v3_complete = (
        ROOT / "workflow" / "rules" / "stage0_v3_cosmetic.smk"
    ).read_text()
    v3_rule = v3_complete.split("rule stage0_v3_complete:", 1)[1]
    assert "rules.compute_skin_score.output.axes" in v3_rule

    producer = (ROOT / "workflow" / "rules" / "stage0_v3_skin.smk").read_text()
    assert '"skin_score.tsv.axes.json"' in producer
