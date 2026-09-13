"""Unit tests for SkinScore composite (INSTRUCTIONS.md §3.1)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage0_skin_score import (  # noqa: E402
    SkinSources,
    compute_skin_score,
    WEIGHTS,
)


def _write_table(dir_: Path, name: str, df: pd.DataFrame) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    df.to_csv(dir_ / name, sep="\t", index=False)


def _write_gene_mapping(hpa_dir: Path, genes: list[str]) -> None:
    _write_table(
        hpa_dir,
        "gene_to_uniprot.tsv",
        pd.DataFrame({"gene": genes, "uniprot": genes}),
    )


@pytest.fixture()
def tmp_sources(tmp_path: Path) -> SkinSources:
    hpa = tmp_path / "hpa"
    proteome = tmp_path / "proteome"
    gtex = tmp_path / "gtex"
    sc = tmp_path / "sc"

    # Skin-enriched proteins: TYR, FLG, MMP1, KRT14 score high on multiple axes.
    # Bulk filler proteins (HOUSEx) score low on every axis.
    skin_genes = ["TYR", "FLG", "MMP1", "KRT14"]
    filler = [f"HOUSE{i}" for i in range(20)]
    _write_gene_mapping(hpa, skin_genes + filler)

    _write_table(hpa, "rna_tissue_consensus.tsv", pd.DataFrame({
        "Gene": skin_genes + filler,
        "Tissue": ["skin"] * (len(skin_genes) + len(filler)),
        "nTPM": [200.0, 150.0, 90.0, 300.0] + [0.5] * len(filler),
    }))
    _write_table(hpa, "rna_single_cell_type.tsv", pd.DataFrame({
        "Gene": skin_genes + filler,
        "Cell type": ["melanocyte", "keratinocyte", "fibroblast", "keratinocyte"]
                      + ["fibroblast"] * len(filler),
        "nTPM": [80.0, 90.0, 70.0, 100.0] + [0.2] * len(filler),
    }))
    _write_table(proteome, "skin_proteome.tsv", pd.DataFrame({
        "uniprot": skin_genes + filler,
        "lfq": [9.0, 8.5, 7.5, 9.2] + [3.0] * len(filler),
    }))
    _write_table(gtex, "skin_tpm.tsv", pd.DataFrame({
        "uniprot": skin_genes + filler,
        "tpm": [120.0, 110.0, 80.0, 150.0] + [1.0] * len(filler),
    }))
    _write_table(sc, "skin_max_celltype.tsv", pd.DataFrame({
        "uniprot": skin_genes + filler,
        "max_expression": [60.0, 70.0, 50.0, 80.0] + [0.5] * len(filler),
    }))

    return SkinSources(hpa, proteome, gtex, sc)


def test_known_skin_proteins_score_above_median(tmp_sources: SkinSources) -> None:
    df = compute_skin_score(tmp_sources)
    median = df["skin_score"].median()
    for gene in ("TYR", "FLG", "MMP1", "KRT14"):
        score = df.loc[df["uniprot"] == gene, "skin_score"].iloc[0]
        assert score > median, f"{gene}: expected > median ({median:.3f}), got {score:.3f}"


def test_hpa_only_sources_produce_skin_score(tmp_path: Path) -> None:
    hpa = tmp_path / "hpa"
    _write_gene_mapping(hpa, ["KRT14", "HOUSE1"])
    _write_table(hpa, "rna_tissue_consensus.tsv", pd.DataFrame({
        "Gene": ["KRT14", "HOUSE1"],
        "Tissue": ["skin", "skin"],
        "nTPM": [100.0, 1.0],
    }))
    _write_table(hpa, "rna_single_cell_type.tsv", pd.DataFrame({
        "Gene": ["KRT14", "HOUSE1"],
        "Cell type": ["keratinocyte", "fibroblast"],
        "nTPM": [80.0, 0.5],
    }))

    df = compute_skin_score(
        SkinSources(
            hpa,
            tmp_path / "missing_proteome",
            tmp_path / "missing_gtex",
            tmp_path / "missing_sc",
        )
    )

    assert set(df["uniprot"]) == {"KRT14", "HOUSE1"}
    assert df.loc[df["uniprot"] == "KRT14", "skin_score"].iloc[0] > 0.0
    assert df.loc[df["uniprot"] == "KRT14", "skin_score"].iloc[0] > (
        df.loc[df["uniprot"] == "HOUSE1", "skin_score"].iloc[0]
    )


def test_hpa_single_cell_accepts_current_ncpm_and_plural_skin_cells(
    tmp_path: Path,
) -> None:
    hpa = tmp_path / "hpa"
    _write_gene_mapping(hpa, ["KRT14", "TYR", "PECAM1", "HOUSE1"])
    _write_table(hpa, "rna_tissue_consensus.tsv", pd.DataFrame({
        "Gene": ["KRT14", "TYR", "PECAM1"],
        "Tissue": ["skin"] * 3,
        "nTPM": [30.0, 40.0, 20.0],
    }))
    _write_table(hpa, "rna_single_cell_type.tsv", pd.DataFrame({
        "Gene": ["KRT14", "TYR", "PECAM1", "HOUSE1"],
        "Gene name": ["KRT14", "TYR", "PECAM1", "HOUSE1"],
        "Cell type": [
            "basal keratinocytes",
            "melanocytes",
            "vascular endothelial cells",
            "adipocytes",
        ],
        "nCPM": [50.0, 70.0, 30.0, 90.0],
    }))

    df = compute_skin_score(
        SkinSources(
            hpa,
            tmp_path / "missing_proteome",
            tmp_path / "missing_gtex",
            tmp_path / "missing_sc",
        )
    )

    assert set(df["uniprot"]) == {"KRT14", "TYR", "PECAM1"}
    assert "HOUSE1" not in set(df["uniprot"])
    assert df["skin_score"].gt(0).any()
    preferred = df.set_index("uniprot")["cell_type_preferred"].to_dict()
    assert preferred == {
        "KRT14": "basal keratinocytes",
        "PECAM1": "vascular endothelial cells",
        "TYR": "melanocytes",
    }


def test_hpa_single_cell_reports_max_positive_cell_type_deterministically(
    tmp_path: Path,
) -> None:
    hpa = tmp_path / "hpa"
    _write_gene_mapping(hpa, ["KRT14", "TYR", "ZERO"])
    _write_table(hpa, "rna_tissue_consensus.tsv", pd.DataFrame({
        "Gene": ["KRT14", "TYR", "ZERO"],
        "Tissue": ["skin"] * 3,
        "nTPM": [30.0, 20.0, 10.0],
    }))
    _write_table(hpa, "rna_single_cell_type.tsv", pd.DataFrame({
        "Gene": ["KRT14", "KRT14", "TYR", "TYR", "ZERO", "ZERO"],
        "Cell type": [
            "keratinocytes",
            "fibroblasts",
            "melanocytes",
            "fibroblasts",
            "keratinocytes",
            "fibroblasts",
        ],
        "nCPM": [10.0, 20.0, 5.0, 5.0, 0.0, 0.0],
    }))

    df = compute_skin_score(
        SkinSources(
            hpa,
            tmp_path / "missing_proteome",
            tmp_path / "missing_gtex",
            tmp_path / "missing_sc",
        )
    )

    preferred = df.set_index("uniprot")["cell_type_preferred"].to_dict()
    assert preferred == {
        "KRT14": "fibroblasts",
        "TYR": "fibroblasts",
        "ZERO": "unknown",
    }


def test_hpa_single_cell_requires_positive_hpa_skin_tissue_support(
    tmp_path: Path,
) -> None:
    hpa = tmp_path / "hpa"
    _write_gene_mapping(hpa, ["SUPPORTED", "ZERO_TISSUE", "CELL_ONLY"])
    _write_table(hpa, "rna_tissue_consensus.tsv", pd.DataFrame({
        "Gene": ["SUPPORTED", "ZERO_TISSUE"],
        "Tissue": ["skin", "skin"],
        "nTPM": [10.0, 0.0],
    }))
    _write_table(hpa, "rna_single_cell_type.tsv", pd.DataFrame({
        "Gene": ["SUPPORTED", "ZERO_TISSUE", "CELL_ONLY"],
        "Cell type": ["keratinocytes", "fibroblasts", "melanocytes"],
        "nCPM": [5.0, 100.0, 100.0],
    }))

    df = compute_skin_score(
        SkinSources(
            hpa,
            tmp_path / "missing_proteome",
            tmp_path / "missing_gtex",
            tmp_path / "missing_sc",
        )
    ).set_index("uniprot")

    assert df.loc["SUPPORTED", "cell_type_preferred"] == "keratinocytes"
    assert df.loc["ZERO_TISSUE", "cell_type_preferred"] == "unknown"
    assert df.loc["CELL_ONLY", "cell_type_preferred"] == "unknown"
    assert df.loc["SUPPORTED", "skin_score"] > df.loc["ZERO_TISSUE", "skin_score"]
    assert df.loc["CELL_ONLY", "skin_score"] == 0.0


def test_hpa_single_cell_alone_does_not_establish_skin_relevance(
    tmp_path: Path,
) -> None:
    hpa = tmp_path / "hpa"
    _write_gene_mapping(hpa, ["KRT14", "TYR"])
    _write_table(hpa, "rna_single_cell_type.tsv", pd.DataFrame({
        "Gene": ["KRT14", "TYR"],
        "Cell type": ["keratinocytes", "melanocytes"],
        "nCPM": [50.0, 70.0],
    }))

    df = compute_skin_score(
        SkinSources(
            hpa,
            tmp_path / "missing_proteome",
            tmp_path / "missing_gtex",
            tmp_path / "missing_sc",
        )
    )

    assert df["skin_score"].eq(0.0).all()
    assert df["cell_type_preferred"].eq("unknown").all()


def test_hpa_gene_ensembl_column_maps_via_proteinatlas(
    tmp_path: Path,
) -> None:
    hpa = tmp_path / "hpa"
    _write_table(hpa, "proteinatlas.tsv", pd.DataFrame({
        "Gene": ["KRT14", "TYR", "HOUSE1"],
        "Ensembl": ["ENSGKRT14", "ENSGTYR", "ENSGHOUSE1"],
        "Uniprot": ["P02533", "P14679", "P00001"],
    }))
    _write_table(hpa, "rna_tissue_consensus.tsv", pd.DataFrame({
        "Gene": ["ENSGKRT14", "ENSGTYR", "ENSGHOUSE1"],
        "Gene name": ["KRT14", "TYR", "HOUSE1"],
        "Tissue": ["skin", "skin", "skin"],
        "nTPM": [100.0, 80.0, 1.0],
    }))
    _write_table(hpa, "rna_single_cell_type.tsv", pd.DataFrame({
        "Gene": ["ENSGKRT14", "ENSGTYR", "ENSGHOUSE1"],
        "Gene name": ["KRT14", "TYR", "HOUSE1"],
        "Cell type": ["keratinocytes", "melanocytes", "adipocytes"],
        "nCPM": [50.0, 60.0, 90.0],
    }))

    df = compute_skin_score(
        SkinSources(
            hpa,
            tmp_path / "missing_proteome",
            tmp_path / "missing_gtex",
            tmp_path / "missing_sc",
        )
    )

    assert set(df["uniprot"]) == {"P02533", "P14679", "P00001"}
    assert df.loc[df["uniprot"] == "P14679", "skin_score"].iloc[0] > 0.0


def test_skin_score_in_unit_interval(tmp_sources: SkinSources) -> None:
    df = compute_skin_score(tmp_sources)
    assert (df["skin_score"] >= 0).all()
    assert (df["skin_score"] <= 1).all()


def test_skin_score_tier_assignment(tmp_sources: SkinSources) -> None:
    df = compute_skin_score(tmp_sources)
    skin_rows = df[df["uniprot"].isin(["TYR", "FLG", "MMP1", "KRT14"])]
    assert (skin_rows["tier"].isin({"medium", "high", "very_high"})).all()


def test_weights_sum_to_one() -> None:
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


# --- A score row with no accession took the whole fast path down ---


_ROOT = Path(__file__).resolve().parents[2]


def test_the_shipped_table_has_an_accession_on_every_row() -> None:
    """A literal "nan" here made stage3_skin_weighting refuse the whole table.

    That step feeds the v3 ranking rule, which is in the fast DAG, so one
    unusable row stopped a fast run from completing at all - and the run that
    predated it reported skin_score 0.000 for MMP1, RXRA and every other
    target that does have a measured value.
    """
    table = _ROOT / "data" / "skin_expression" / "skin_score.tsv"
    if not table.is_file():
        pytest.skip("Stage 0 skin score table has not been built")
    frame = pd.read_csv(table, sep="\t")

    blank = frame[
        frame["uniprot"].isna() | (frame["uniprot"].astype(str).str.strip() == "")
    ]
    assert blank.empty, f"rows with no accession: {blank.index.tolist()[:5]}"
    assert not frame["uniprot"].duplicated().any()
    # And the values downstream actually reads are present.
    indexed = frame.set_index("uniprot")
    for accession in ("P03956", "P19793", "P14679"):
        assert accession in indexed.index
        assert float(indexed.loc[accession, "skin_score"]) > 0


def test_the_generator_refuses_to_emit_a_row_without_an_accession() -> None:
    """Fail closed at the source, not just in the consumer."""
    source = (_ROOT / "scripts" / "stage0_skin_score.py").read_text(encoding="utf-8")
    body = source.split("def main(", 1)[1]
    assert "SkinScore output has rows with no UniProt accession" in body
    assert "SkinScore output has duplicate accessions" in body
