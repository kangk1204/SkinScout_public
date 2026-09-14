"""Regression tests for Stage 3 disagreement input gates."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage3_disagreement import rank_targets  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    docking_top = tmp_path / "top50.csv"
    autodock = tmp_path / "autodock.tsv"
    psichic = tmp_path / "psichic.tsv"
    chembl = tmp_path / "chembl"
    out = tmp_path / "disagreement.json"
    chembl.mkdir()
    pd.DataFrame([{"target_id": "P1"}, {"target_id": "P2"}]).to_csv(
        docking_top,
        index=False,
    )
    pd.DataFrame([
        {"target_id": "P1", "neg_vina_score": 9.0},
        {"target_id": "P2", "neg_vina_score": 1.0},
    ]).to_csv(autodock, sep="\t", index=False)
    pd.DataFrame([
        {"target_id": "P1", "psichic_score": 0.9},
        {"target_id": "P2", "psichic_score": 0.1},
    ]).to_csv(psichic, sep="\t", index=False)
    pd.DataFrame([
        {"uniprot": "P1", "class": "kinase"},
        {"uniprot": "P2", "class": "gpcr"},
    ]).to_parquet(chembl / "target_classes.parquet")
    out.write_text("stale\n")
    return docking_top, autodock, psichic, chembl, out


def run_disagreement(
    docking_top: Path,
    autodock: Path,
    psichic: Path,
    chembl: Path,
    out: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "stage3_disagreement.py"),
            "--docking-top", str(docking_top),
            "--autodock-full", str(autodock),
            "--psichic", str(psichic),
            "--chembl-dir", str(chembl),
            "--out-json", str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_disagreement_requires_target_class_reference(tmp_path: Path) -> None:
    docking_top, autodock, psichic, chembl, out = write_inputs(tmp_path)
    (chembl / "target_classes.parquet").unlink()

    res = run_disagreement(docking_top, autodock, psichic, chembl, out)

    assert res.returncode != 0
    assert "Target class reference is required" in res.stderr
    assert not out.exists()


def test_disagreement_rejects_empty_psichic_scores(tmp_path: Path) -> None:
    docking_top, autodock, psichic, chembl, out = write_inputs(tmp_path)
    psichic.write_text("target_id\tpsichic_score\n")

    res = run_disagreement(docking_top, autodock, psichic, chembl, out)

    assert res.returncode != 0
    assert "PSICHIC score table contains no rows" in res.stderr
    assert not out.exists()


def test_disagreement_rejects_missing_psichic_score_column(tmp_path: Path) -> None:
    docking_top, autodock, psichic, chembl, out = write_inputs(tmp_path)
    pd.DataFrame([{"target_id": "P1", "score": 0.9}]).to_csv(
        psichic,
        sep="\t",
        index=False,
    )

    res = run_disagreement(docking_top, autodock, psichic, chembl, out)

    assert res.returncode != 0
    assert "PSICHIC score table missing required columns" in res.stderr
    assert not out.exists()


def test_disagreement_rejects_partially_invalid_autodock_scores(
    tmp_path: Path,
) -> None:
    docking_top, autodock, psichic, chembl, out = write_inputs(tmp_path)
    autodock.write_text("target_id\tneg_vina_score\nP1\t9.0\nP2\tnot-a-number\n")

    res = run_disagreement(docking_top, autodock, psichic, chembl, out)

    assert res.returncode != 0
    assert (
        "AutoDock full score table column 'neg_vina_score' contains invalid values"
        in res.stderr
    )
    assert not out.exists()


def test_disagreement_rejects_boolean_psichic_scores(tmp_path: Path) -> None:
    docking_top, autodock, psichic, chembl, out = write_inputs(tmp_path)
    psichic.write_text("target_id\tpsichic_score\nP1\tTrue\nP2\tFalse\n")

    res = run_disagreement(docking_top, autodock, psichic, chembl, out)

    assert res.returncode != 0
    assert (
        "PSICHIC score table column 'psichic_score' contains invalid values"
        in res.stderr
    )
    assert not out.exists()


def test_disagreement_rejects_blank_psichic_target_ids(tmp_path: Path) -> None:
    docking_top, autodock, psichic, chembl, out = write_inputs(tmp_path)
    psichic.write_text("target_id\tpsichic_score\nP1\t0.9\n \t0.1\n")

    res = run_disagreement(docking_top, autodock, psichic, chembl, out)

    assert res.returncode != 0
    assert "PSICHIC score table column 'target_id' contains blank values" in res.stderr
    assert not out.exists()


def test_disagreement_rejects_duplicate_autodock_target_ids(
    tmp_path: Path,
) -> None:
    docking_top, autodock, psichic, chembl, out = write_inputs(tmp_path)
    autodock.write_text("target_id\tneg_vina_score\nP1\t9.0\nP1\t8.0\n")

    res = run_disagreement(docking_top, autodock, psichic, chembl, out)

    assert res.returncode != 0
    assert "AutoDock full score table contains duplicate target_id values: P1" in (
        res.stderr
    )
    assert not out.exists()


def test_disagreement_rejects_duplicate_docking_top_targets(
    tmp_path: Path,
) -> None:
    docking_top, autodock, psichic, chembl, out = write_inputs(tmp_path)
    pd.DataFrame([{"target_id": "P1"}, {"target_id": "P1"}]).to_csv(
        docking_top,
        index=False,
    )

    res = run_disagreement(docking_top, autodock, psichic, chembl, out)

    assert res.returncode != 0
    assert "Docking top consensus contains duplicate target_id values: P1" in (
        res.stderr
    )
    assert not out.exists()


def test_disagreement_rejects_docking_top_target_missing_from_autodock_full(
    tmp_path: Path,
) -> None:
    docking_top, autodock, psichic, chembl, out = write_inputs(tmp_path)
    pd.DataFrame([{"target_id": "P3"}, {"target_id": "P2"}]).to_csv(
        docking_top,
        index=False,
    )
    pd.DataFrame(
        [
            {"target_id": "P1", "neg_vina_score": 9.0},
            {"target_id": "P2", "neg_vina_score": 5.0},
        ]
    ).to_csv(autodock, sep="\t", index=False)
    pd.DataFrame(
        [
            {"target_id": "P3", "psichic_score": 0.9},
            {"target_id": "P2", "psichic_score": 0.8},
            {"target_id": "P1", "psichic_score": 0.1},
        ]
    ).to_csv(psichic, sep="\t", index=False)
    pd.DataFrame(
        [
            {"uniprot": "P1", "class": "kinase"},
            {"uniprot": "P2", "class": "gpcr"},
            {"uniprot": "P3", "class": "protease"},
        ]
    ).to_parquet(chembl / "target_classes.parquet")

    res = run_disagreement(docking_top, autodock, psichic, chembl, out)

    assert res.returncode != 0
    assert (
        "Docking top consensus target_id value(s) missing from "
        "AutoDock full score table: P3"
    ) in res.stderr
    assert not out.exists()


def test_disagreement_rejects_docking_top_target_missing_from_psichic_scores(
    tmp_path: Path,
) -> None:
    docking_top, autodock, psichic, chembl, out = write_inputs(tmp_path)
    pd.DataFrame([{"target_id": "P3"}, {"target_id": "P2"}]).to_csv(
        docking_top,
        index=False,
    )
    pd.DataFrame(
        [
            {"target_id": "P1", "neg_vina_score": 9.0},
            {"target_id": "P2", "neg_vina_score": 5.0},
            {"target_id": "P3", "neg_vina_score": 1.0},
        ]
    ).to_csv(autodock, sep="\t", index=False)
    pd.DataFrame(
        [
            {"target_id": "P1", "psichic_score": 0.9},
            {"target_id": "P2", "psichic_score": 0.8},
        ]
    ).to_csv(psichic, sep="\t", index=False)
    pd.DataFrame(
        [
            {"uniprot": "P1", "class": "kinase"},
            {"uniprot": "P2", "class": "gpcr"},
            {"uniprot": "P3", "class": "protease"},
        ]
    ).to_parquet(chembl / "target_classes.parquet")

    res = run_disagreement(docking_top, autodock, psichic, chembl, out)

    assert res.returncode != 0
    assert (
        "Docking top consensus target_id value(s) missing from "
        "PSICHIC score table: P3"
    ) in res.stderr
    assert not out.exists()


def test_disagreement_rejects_blank_target_class_uniprots(tmp_path: Path) -> None:
    docking_top, autodock, psichic, chembl, out = write_inputs(tmp_path)
    pd.DataFrame([
        {"uniprot": "P1", "class": "kinase"},
        {"uniprot": " ", "class": "gpcr"},
    ]).to_parquet(chembl / "target_classes.parquet")

    res = run_disagreement(docking_top, autodock, psichic, chembl, out)

    assert res.returncode != 0
    assert (
        "Target class reference column 'uniprot' contains blank values" in res.stderr
    )
    assert not out.exists()


def test_disagreement_rejects_blank_target_class_labels(tmp_path: Path) -> None:
    docking_top, autodock, psichic, chembl, out = write_inputs(tmp_path)
    pd.DataFrame([
        {"uniprot": "P1", "class": "kinase"},
        {"uniprot": "P2", "class": " "},
    ]).to_parquet(chembl / "target_classes.parquet")

    res = run_disagreement(docking_top, autodock, psichic, chembl, out)

    assert res.returncode != 0
    assert "Target class reference column 'class' contains blank values" in res.stderr
    assert not out.exists()


def test_disagreement_rejects_duplicate_target_class_uniprots(tmp_path: Path) -> None:
    docking_top, autodock, psichic, chembl, out = write_inputs(tmp_path)
    pd.DataFrame([
        {"uniprot": "P1", "class": "kinase"},
        {"uniprot": "P1", "class": "gpcr"},
    ]).to_parquet(chembl / "target_classes.parquet")

    res = run_disagreement(docking_top, autodock, psichic, chembl, out)

    assert res.returncode != 0
    assert (
        "Target class reference contains duplicate uniprot values: P1" in res.stderr
    )
    assert not out.exists()


def test_disagreement_ranks_numeric_scores_not_lexicographic(tmp_path: Path) -> None:
    df = pd.DataFrame(
        [
            {"target_id": "P1", "neg_vina_score": "9"},
            {"target_id": "P2", "neg_vina_score": "10"},
        ]
    )

    assert rank_targets(df, "neg_vina_score", "AutoDock full score table") == [
        "P2",
        "P1",
    ]


def test_disagreement_uses_docking_top_consensus_for_overlap(
    tmp_path: Path,
) -> None:
    docking_top, autodock, psichic, chembl, out = write_inputs(tmp_path)
    pd.DataFrame([{"target_id": "P3"}, {"target_id": "P2"}]).to_csv(
        docking_top,
        index=False,
    )
    pd.DataFrame(
        [
            {"target_id": "P1", "neg_vina_score": 9.0},
            {"target_id": "P2", "neg_vina_score": 5.0},
            {"target_id": "P3", "neg_vina_score": 1.0},
        ]
    ).to_csv(autodock, sep="\t", index=False)
    pd.DataFrame(
        [
            {"target_id": "P3", "psichic_score": 0.9},
            {"target_id": "P2", "psichic_score": 0.8},
            {"target_id": "P1", "psichic_score": 0.1},
        ]
    ).to_csv(psichic, sep="\t", index=False)
    pd.DataFrame(
        [
            {"uniprot": "P1", "class": "kinase"},
            {"uniprot": "P2", "class": "gpcr"},
            {"uniprot": "P3", "class": "protease"},
        ]
    ).to_parquet(chembl / "target_classes.parquet")

    res = run_disagreement(docking_top, autodock, psichic, chembl, out)

    assert res.returncode == 0, res.stderr
    payload = json.loads(out.read_text())
    assert payload["both_top"] == ["P2", "P3"]
    assert payload["overlap_at_top_n"] == 2


def test_disagreement_writes_payload_for_valid_inputs(tmp_path: Path) -> None:
    docking_top, autodock, psichic, chembl, out = write_inputs(tmp_path)

    res = run_disagreement(docking_top, autodock, psichic, chembl, out)

    assert res.returncode == 0, res.stderr
    payload = json.loads(out.read_text())
    assert payload["n_docking_full"] == 2
    assert payload["n_dti_full"] == 2
    assert payload["both_top"] == ["P1", "P2"]
    assert payload["per_class"]["both"] == {"gpcr": 1, "kinase": 1}
