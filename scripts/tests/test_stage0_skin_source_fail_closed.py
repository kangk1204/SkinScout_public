"""Regression tests for Stage 0 skin-expression source gates."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def run_script(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def write_hpa_gene_mapping(hpa_dir: Path, genes: list[str]) -> None:
    hpa_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"gene": genes, "uniprot": genes}).to_csv(
        hpa_dir / "gene_to_uniprot.tsv",
        sep="\t",
        index=False,
    )


def test_skin_proteome_missing_raw_fails_by_default(tmp_path: Path) -> None:
    out_dir = tmp_path / "proteome"
    out_dir.mkdir()
    (out_dir / "skin_proteome.tsv").write_text("stale\n")
    res = run_script(["scripts/stage0_skin_proteome.py", "--out-dir", str(out_dir)])

    assert res.returncode != 0
    assert "supplementary LFQ TSV is required" in res.stderr
    assert not (out_dir / "skin_proteome.tsv").exists()


def test_skin_proteome_placeholder_requires_explicit_flag(tmp_path: Path) -> None:
    out_dir = tmp_path / "proteome"
    res = run_script([
        "scripts/stage0_skin_proteome.py",
        "--out-dir", str(out_dir),
        "--allow-empty-placeholder",
    ])

    assert res.returncode == 0, res.stderr
    assert (out_dir / "skin_proteome.tsv").read_text() == "uniprot\tlfq\n"


def test_gtex_missing_gct_fails_by_default(tmp_path: Path) -> None:
    out_dir = tmp_path / "gtex"
    out_dir.mkdir()
    (out_dir / "skin_tpm.tsv").write_text("stale\n")
    res = run_script(["scripts/stage0_gtex_skin.py", "--out-dir", str(out_dir)])

    assert res.returncode != 0
    assert "GTEx GCT is required" in res.stderr
    assert not (out_dir / "skin_tpm.tsv").exists()


def test_gtex_placeholder_requires_explicit_flag(tmp_path: Path) -> None:
    out_dir = tmp_path / "gtex"
    res = run_script([
        "scripts/stage0_gtex_skin.py",
        "--out-dir", str(out_dir),
        "--allow-empty-placeholder",
    ])

    assert res.returncode == 0, res.stderr
    assert (out_dir / "skin_tpm.tsv").read_text() == "uniprot\ttpm\n"


def test_skin_score_empty_sources_fail_without_diagnostic_flag(tmp_path: Path) -> None:
    out = tmp_path / "skin_score.tsv"
    out.write_text("stale\n")
    res = run_script([
        "scripts/stage0_skin_score.py",
        "--hpa-dir", str(tmp_path / "hpa"),
        "--proteome-dir", str(tmp_path / "proteome"),
        "--gtex-dir", str(tmp_path / "gtex"),
        "--sc-dir", str(tmp_path / "sc"),
        "--out-tsv", str(out),
    ])

    assert res.returncode != 0
    assert "SkinScore has no evaluable rows" in res.stderr
    assert not out.exists()


def test_skin_score_rejects_present_source_with_bad_schema(tmp_path: Path) -> None:
    hpa = tmp_path / "hpa"
    hpa.mkdir()
    write_hpa_gene_mapping(hpa, ["KRT14"])
    pd.DataFrame([
        {"Gene": "KRT14", "Tissue": "skin", "nTPM": 100.0},
    ]).to_csv(hpa / "rna_tissue_consensus.tsv", sep="\t", index=False)
    gtex = tmp_path / "gtex"
    gtex.mkdir()
    pd.DataFrame([
        {"uniprot": "KRT14", "wrong_value": 20.0},
    ]).to_csv(gtex / "skin_tpm.tsv", sep="\t", index=False)
    out = tmp_path / "skin_score.tsv"
    out.write_text("stale\n")

    res = run_script([
        "scripts/stage0_skin_score.py",
        "--hpa-dir", str(hpa),
        "--proteome-dir", str(tmp_path / "proteome"),
        "--gtex-dir", str(gtex),
        "--sc-dir", str(tmp_path / "sc"),
        "--out-tsv", str(out),
    ])

    assert res.returncode != 0
    assert "skin_tpm.tsv is missing required columns" in res.stderr
    assert not out.exists()


def test_skin_score_rejects_blank_source_uniprot_and_removes_stale_output(
    tmp_path: Path,
) -> None:
    gtex = tmp_path / "gtex"
    gtex.mkdir()
    pd.DataFrame([
        {"uniprot": " ", "tpm": 20.0},
    ]).to_csv(gtex / "skin_tpm.tsv", sep="\t", index=False)
    out = tmp_path / "skin_score.tsv"
    out.write_text("stale\n")

    res = run_script([
        "scripts/stage0_skin_score.py",
        "--hpa-dir", str(tmp_path / "hpa"),
        "--proteome-dir", str(tmp_path / "proteome"),
        "--gtex-dir", str(gtex),
        "--sc-dir", str(tmp_path / "sc"),
        "--out-tsv", str(out),
    ])

    assert res.returncode != 0
    assert "GTEx skin source column 'uniprot' contains blank values" in res.stderr
    assert not out.exists()


def test_skin_score_rejects_blank_gtex_tpm_and_removes_stale_output(
    tmp_path: Path,
) -> None:
    gtex = tmp_path / "gtex"
    gtex.mkdir()
    pd.DataFrame([
        {"uniprot": "KRT14", "tpm": ""},
    ]).to_csv(gtex / "skin_tpm.tsv", sep="\t", index=False)
    out = tmp_path / "skin_score.tsv"
    out.write_text("stale\n")

    res = run_script([
        "scripts/stage0_skin_score.py",
        "--hpa-dir", str(tmp_path / "hpa"),
        "--proteome-dir", str(tmp_path / "proteome"),
        "--gtex-dir", str(gtex),
        "--sc-dir", str(tmp_path / "sc"),
        "--out-tsv", str(out),
    ])

    assert res.returncode != 0
    assert "GTEx skin source column 'tpm' contains blank values" in res.stderr
    assert not out.exists()


def test_skin_score_rejects_blank_proteome_lfq_and_removes_stale_output(
    tmp_path: Path,
) -> None:
    proteome = tmp_path / "proteome"
    proteome.mkdir()
    pd.DataFrame([
        {"uniprot": "KRT14", "lfq": ""},
    ]).to_csv(proteome / "skin_proteome.tsv", sep="\t", index=False)
    out = tmp_path / "skin_score.tsv"
    out.write_text("stale\n")

    res = run_script([
        "scripts/stage0_skin_score.py",
        "--hpa-dir", str(tmp_path / "hpa"),
        "--proteome-dir", str(proteome),
        "--gtex-dir", str(tmp_path / "gtex"),
        "--sc-dir", str(tmp_path / "sc"),
        "--out-tsv", str(out),
    ])

    assert res.returncode != 0
    assert "Skin proteome source column 'value' contains blank values" in res.stderr
    assert not out.exists()


def test_skin_score_rejects_blank_single_cell_expression_and_removes_stale_output(
    tmp_path: Path,
) -> None:
    sc = tmp_path / "sc"
    sc.mkdir()
    pd.DataFrame([
        {"uniprot": "KRT14", "max_expression": ""},
    ]).to_csv(sc / "skin_max_celltype.tsv", sep="\t", index=False)
    out = tmp_path / "skin_score.tsv"
    out.write_text("stale\n")

    res = run_script([
        "scripts/stage0_skin_score.py",
        "--hpa-dir", str(tmp_path / "hpa"),
        "--proteome-dir", str(tmp_path / "proteome"),
        "--gtex-dir", str(tmp_path / "gtex"),
        "--sc-dir", str(sc),
        "--out-tsv", str(out),
    ])

    assert res.returncode != 0
    assert "Single-cell skin source column 'max_expression' contains blank values" in res.stderr
    assert not out.exists()


def test_skin_score_rejects_blank_hpa_tissue_expression_and_removes_stale_output(
    tmp_path: Path,
) -> None:
    hpa = tmp_path / "hpa"
    hpa.mkdir()
    write_hpa_gene_mapping(hpa, ["KRT14"])
    pd.DataFrame([
        {"Gene": "KRT14", "Tissue": "skin", "nTPM": ""},
    ]).to_csv(hpa / "rna_tissue_consensus.tsv", sep="\t", index=False)
    out = tmp_path / "skin_score.tsv"
    out.write_text("stale\n")

    res = run_script([
        "scripts/stage0_skin_score.py",
        "--hpa-dir", str(hpa),
        "--proteome-dir", str(tmp_path / "proteome"),
        "--gtex-dir", str(tmp_path / "gtex"),
        "--sc-dir", str(tmp_path / "sc"),
        "--out-tsv", str(out),
    ])

    assert res.returncode != 0
    assert "HPA tissue source column 'ntpm' contains blank values" in res.stderr
    assert not out.exists()


def test_skin_score_rejects_blank_hpa_cell_expression_and_removes_stale_output(
    tmp_path: Path,
) -> None:
    hpa = tmp_path / "hpa"
    hpa.mkdir()
    write_hpa_gene_mapping(hpa, ["KRT14"])
    pd.DataFrame([
        {"Gene": "KRT14", "cell_type": "keratinocyte", "nTPM": ""},
    ]).to_csv(hpa / "rna_single_cell_type.tsv", sep="\t", index=False)
    out = tmp_path / "skin_score.tsv"
    out.write_text("stale\n")

    res = run_script([
        "scripts/stage0_skin_score.py",
        "--hpa-dir", str(hpa),
        "--proteome-dir", str(tmp_path / "proteome"),
        "--gtex-dir", str(tmp_path / "gtex"),
        "--sc-dir", str(tmp_path / "sc"),
        "--out-tsv", str(out),
    ])

    assert res.returncode != 0
    assert "HPA cell source column 'ntpm' contains blank values" in res.stderr
    assert not out.exists()


def test_skin_score_v2_missing_proteinatlas_removes_stale_output(tmp_path: Path) -> None:
    out = tmp_path / "skin_score_v2.tsv"
    out.write_text("stale\n")

    res = run_script([
        "scripts/stage0_skin_score_v2.py",
        "--proteinatlas-tsv", str(tmp_path / "missing_proteinatlas.tsv"),
        "--out-tsv", str(out),
    ])

    assert res.returncode != 0
    assert "missing_proteinatlas.tsv" in res.stderr
    assert not out.exists()


def test_skin_score_v2_rejects_malformed_expression_values(tmp_path: Path) -> None:
    proteinatlas = tmp_path / "proteinatlas.tsv"
    pd.DataFrame([
        {
            "Uniprot": "P02533",
            "Gene": "KRT14",
            "RNA tissue specific nTPM": "Skin: not-a-number",
            "RNA single cell type specific nCPM": "Keratinocytes: 12.0",
            "RNA tissue specificity": "Tissue enriched",
            "RNA tissue distribution": "Skin",
        },
    ]).to_csv(proteinatlas, sep="\t", index=False)
    out = tmp_path / "skin_score_v2.tsv"
    out.write_text("stale\n")

    res = run_script([
        "scripts/stage0_skin_score_v2.py",
        "--proteinatlas-tsv", str(proteinatlas),
        "--out-tsv", str(out),
    ])

    assert res.returncode != 0
    assert "RNA tissue specific nTPM contains non-numeric value" in res.stderr
    assert "not-a-number" in res.stderr
    assert not out.exists()


def test_skin_score_v2_rejects_blank_expression_cells_and_removes_stale_output(
    tmp_path: Path,
) -> None:
    proteinatlas = tmp_path / "proteinatlas.tsv"
    pd.DataFrame([
        {
            "Uniprot": "P02533",
            "Gene": "KRT14",
            "RNA tissue specific nTPM": "",
            "RNA single cell type specific nCPM": "Keratinocytes: 12.0",
            "RNA tissue specificity": "Tissue enriched",
            "RNA tissue distribution": "Skin",
        },
    ]).to_csv(proteinatlas, sep="\t", index=False)
    out = tmp_path / "skin_score_v2.tsv"
    out.write_text("stale\n")

    res = run_script([
        "scripts/stage0_skin_score_v2.py",
        "--proteinatlas-tsv", str(proteinatlas),
        "--out-tsv", str(out),
    ])

    assert res.returncode == 0, res.stderr
    scored = pd.read_csv(out, sep="\t")
    assert set(scored["uniprot"]) == {"P02533"}
    assert "stale" not in out.read_text()


def test_skin_score_v2_rejects_blank_uniprot_and_removes_stale_output(
    tmp_path: Path,
) -> None:
    proteinatlas = tmp_path / "proteinatlas.tsv"
    pd.DataFrame([
        {
            "Uniprot": " ",
            "Gene": "KRT14",
            "RNA tissue specific nTPM": "Skin: 10.0",
            "RNA single cell type specific nCPM": "Keratinocytes: 12.0",
            "RNA tissue specificity": "Tissue enriched",
            "RNA tissue distribution": "Skin",
        },
    ]).to_csv(proteinatlas, sep="\t", index=False)
    out = tmp_path / "skin_score_v2.tsv"
    out.write_text("stale\n")

    res = run_script([
        "scripts/stage0_skin_score_v2.py",
        "--proteinatlas-tsv", str(proteinatlas),
        "--out-tsv", str(out),
    ])

    assert res.returncode != 0
    assert "proteinatlas.tsv column 'Uniprot' contains blank values" in res.stderr
    assert not out.exists()


def test_skin_score_v2_rejects_duplicate_uniprot_and_removes_stale_output(
    tmp_path: Path,
) -> None:
    proteinatlas = tmp_path / "proteinatlas.tsv"
    pd.DataFrame([
        {
            "Uniprot": "P02533",
            "Gene": "KRT14",
            "RNA tissue specific nTPM": "Skin: 10.0",
            "RNA single cell type specific nCPM": "Keratinocytes: 12.0",
            "RNA tissue specificity": "Tissue enriched",
            "RNA tissue distribution": "Skin",
        },
        {
            "Uniprot": "P02533",
            "Gene": "KRT14_ALT",
            "RNA tissue specific nTPM": "Skin: 20.0",
            "RNA single cell type specific nCPM": "Keratinocytes: 24.0",
            "RNA tissue specificity": "Tissue enriched",
            "RNA tissue distribution": "Skin",
        },
    ]).to_csv(proteinatlas, sep="\t", index=False)
    out = tmp_path / "skin_score_v2.tsv"
    out.write_text("stale\n")

    res = run_script([
        "scripts/stage0_skin_score_v2.py",
        "--proteinatlas-tsv", str(proteinatlas),
        "--out-tsv", str(out),
    ])

    assert res.returncode != 0
    assert "proteinatlas.tsv contains duplicate primary Uniprot values" in res.stderr
    assert "P02533" in res.stderr
    assert not out.exists()


def test_skin_score_writes_when_at_least_one_axis_has_rows(tmp_path: Path) -> None:
    hpa = tmp_path / "hpa"
    hpa.mkdir()
    write_hpa_gene_mapping(hpa, ["KRT14", "HOUSE"])
    pd.DataFrame([
        {"Gene": "KRT14", "Tissue": "skin", "nTPM": 100.0},
        {"Gene": "HOUSE", "Tissue": "skin", "nTPM": 1.0},
    ]).to_csv(hpa / "rna_tissue_consensus.tsv", sep="\t", index=False)
    out = tmp_path / "skin_score.tsv"

    res = run_script([
        "scripts/stage0_skin_score.py",
        "--hpa-dir", str(hpa),
        "--proteome-dir", str(tmp_path / "proteome"),
        "--gtex-dir", str(tmp_path / "gtex"),
        "--sc-dir", str(tmp_path / "sc"),
        "--out-tsv", str(out),
    ])

    assert res.returncode == 0, res.stderr
    assert set(pd.read_csv(out, sep="\t")["uniprot"]) == {"KRT14", "HOUSE"}
