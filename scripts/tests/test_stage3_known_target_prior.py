from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

try:
    from rdkit import Chem
except ModuleNotFoundError:  # pragma: no cover
    Chem = None

if Chem is None:
    pytest.skip("RDKit is required for known-target prior tests", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "stage3_known_target_prior.py"


def run_prior(
    tmp_path: Path,
    compound_json: Path,
    panel_csv: Path,
    args: list[str] | None = None,
    supplemental_panel: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    out_csv = tmp_path / "known_target_priors.csv"
    command = [
        sys.executable,
        str(SCRIPT),
        "--compound-json",
        str(compound_json),
        "--known-panel",
        str(panel_csv),
        "--out-csv",
        str(out_csv),
        "--enabled",
        "true",
    ]
    if args:
        command.extend(args)
    if supplemental_panel is not None:
        command.extend(["--supplemental-panels", str(supplemental_panel)])
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )


def write_compound_json(path: Path, smiles: str) -> Path:
    path.write_text(f'{{"canonical_smiles": "{smiles}"}}\n')
    return path


def test_stage3_known_target_prior_exact_match_stays_full_score(tmp_path: Path) -> None:
    compound = write_compound_json(tmp_path / "compound.json", "CCOC")
    pd.DataFrame([
        {
            "case_id": "test_exact",
            "smiles": "COCC",  # same after canonicalization
            "known_targets": "P12345;P67890",
            "known_target_labels": "T1;T2",
        }
    ]).to_csv(tmp_path / "panel.csv", index=False)

    res = run_prior(tmp_path, compound, tmp_path / "panel.csv")

    assert res.returncode == 0, res.stderr
    out = pd.read_csv(tmp_path / "known_target_priors.csv")
    assert out.shape[0] == 2
    assert set(out["target_id"]) == {"P12345", "P67890"}
    assert set(out["prior_score"]) == {1.0}
    assert set(out["source_panel"]) == {"skin_known_target_panel"}
    assert set(out["source_case_id"]) == {"test_exact"}


def test_stage3_known_target_prior_similarity_fallback_can_rescue_non_exact_match(tmp_path: Path) -> None:
    compound = write_compound_json(tmp_path / "compound.json", "CCO")
    pd.DataFrame([
        {
            "case_id": "ethanol_similar",
            "smiles": "CCCO",  # close neighbor, not exact
            "known_targets": "P11111",
            "known_target_labels": "X",
        }
    ]).to_csv(tmp_path / "panel.csv", index=False)

    res = run_prior(
        tmp_path,
        compound,
        tmp_path / "panel.csv",
        args=[
            "--similarity-enabled",
            "true",
            "--similarity-threshold",
            "0.01",
        ],
    )

    assert res.returncode == 0, res.stderr
    out = pd.read_csv(tmp_path / "known_target_priors.csv")
    assert out.shape[0] == 1
    assert out.loc[0, "target_id"] == "P11111"
    assert out.loc[0, "prior_score"] > 0.0
    assert out.loc[0, "prior_score"] <= 1.0


def test_stage3_known_target_prior_high_similarity_threshold_blocks_weak_match(tmp_path: Path) -> None:
    compound = write_compound_json(tmp_path / "compound.json", "CCO")
    pd.DataFrame([
        {
            "case_id": "ethanol_not_match",
            "smiles": "CCCO",
            "known_targets": "P11111",
            "known_target_labels": "X",
        }
    ]).to_csv(tmp_path / "panel.csv", index=False)

    res = run_prior(
        tmp_path,
        compound,
        tmp_path / "panel.csv",
        args=[
            "--similarity-enabled",
            "true",
            "--similarity-threshold",
            "0.99",
        ],
    )

    assert res.returncode == 0, res.stderr
    out = pd.read_csv(tmp_path / "known_target_priors.csv")
    assert out.empty


def test_stage3_known_target_prior_prefers_exact_when_similarity_available(tmp_path: Path) -> None:
    compound = write_compound_json(tmp_path / "compound.json", "C1=CC=CC=C1O")
    pd.DataFrame([
        {
            "case_id": "phenol_exact",
            "smiles": "OC1=CC=CC=C1",
            "known_targets": "P1",
            "known_target_labels": "target_exact",
        },
        {
            "case_id": "phenol_like",
            "smiles": "CC1=CC=CC=C1",  # less similar aromatic analog
            "known_targets": "P2",
            "known_target_labels": "target_like",
        },
    ]).to_csv(tmp_path / "panel.csv", index=False)

    res = run_prior(
        tmp_path,
        compound,
        tmp_path / "panel.csv",
        args=[
            "--similarity-enabled",
            "true",
            "--similarity-threshold",
            "0.05",
        ],
    )

    assert res.returncode == 0, res.stderr
    out = pd.read_csv(tmp_path / "known_target_priors.csv")
    assert not out.empty
    assert set(out["target_id"]) == {"P1"}
    exact = out.set_index("target_id").loc["P1"]
    assert exact["prior_score"] == 1.0


def test_stage3_known_target_prior_exact_inchikey_excludes_similarity_fallback(
    tmp_path: Path,
) -> None:
    compound = write_compound_json(tmp_path / "compound.json", "CCO")
    pd.DataFrame([
        {
            "case_id": "ethanol_exact_inchi",
            "smiles": "CCCO",
            "inchi_key": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
            "known_targets": "P1",
        },
        {
            "case_id": "similar_neighbor",
            "smiles": "CCCO",
            "known_targets": "P2",
        },
    ]).to_csv(tmp_path / "panel.csv", index=False)

    res = run_prior(
        tmp_path,
        compound,
        tmp_path / "panel.csv",
        args=[
            "--similarity-enabled",
            "true",
            "--similarity-threshold",
            "0.01",
        ],
    )

    assert res.returncode == 0, res.stderr
    out = pd.read_csv(tmp_path / "known_target_priors.csv")
    assert set(out["target_id"]) == {"P1"}


def test_stage3_known_target_prior_supplemental_panel_and_source_weight(tmp_path: Path) -> None:
    compound = write_compound_json(tmp_path / "compound.json", "CCO")
    pd.DataFrame([
        {
            "case_id": "exact_main",
            "smiles": "CCO",
            "known_targets": "P1",
            "source_weight": "0.2",
        }
    ]).to_csv(tmp_path / "panel.csv", index=False)
    pd.DataFrame([
        {
            "case_id": "supplemental_similar",
            "smiles": "CCCO",
            "known_targets": "P2",
            "source_weight": "0.9",
        }
    ]).to_csv(tmp_path / "supplemental.csv", index=False)

    res = run_prior(
        tmp_path,
        compound,
        tmp_path / "panel.csv",
        supplemental_panel=tmp_path / "supplemental.csv",
        args=[
            "--similarity-enabled",
            "true",
            "--similarity-threshold",
            "0.01",
            "--max-similarity-candidates",
            "1",
        ],
    )

    assert res.returncode == 0, res.stderr
    out = pd.read_csv(tmp_path / "known_target_priors.csv")
    assert set(out["target_id"]) == {"P1"}
    assert out.loc[out["target_id"] == "P1", "prior_score"].iat[0] == 0.2


def test_stage3_known_target_prior_limits_similarity_candidates(tmp_path: Path) -> None:
    compound = write_compound_json(tmp_path / "compound.json", "CCNCC")
    pd.DataFrame([
        {
            "case_id": "case_1",
            "smiles": "CCOC",
            "known_targets": "P1",
            "source_weight": "1",
        },
        {
            "case_id": "case_2",
            "smiles": "CCN",
            "known_targets": "P2",
            "source_weight": "1",
        },
    ]).to_csv(tmp_path / "panel.csv", index=False)

    res = run_prior(
        tmp_path,
        compound,
        tmp_path / "panel.csv",
        args=[
            "--similarity-enabled",
            "true",
            "--similarity-threshold",
            "0.01",
            "--max-similarity-candidates",
            "1",
        ],
    )

    assert res.returncode == 0, res.stderr
    out = pd.read_csv(tmp_path / "known_target_priors.csv")
    assert len(out) == 1


def test_stage3_known_target_prior_similarity_candidate_limit_keeps_all_row_targets(
    tmp_path: Path,
) -> None:
    compound = write_compound_json(tmp_path / "compound.json", "CCNCC")
    pd.DataFrame([
        {
            "case_id": "best_multi_target_row",
            "smiles": "CCN",
            "known_targets": "P1;P2",
            "source_weight": "1",
        },
        {
            "case_id": "second_row",
            "smiles": "CCOC",
            "known_targets": "P3",
            "source_weight": "1",
        },
    ]).to_csv(tmp_path / "panel.csv", index=False)

    res = run_prior(
        tmp_path,
        compound,
        tmp_path / "panel.csv",
        args=[
            "--similarity-enabled",
            "true",
            "--similarity-threshold",
            "0.01",
            "--max-similarity-candidates",
            "1",
        ],
    )

    assert res.returncode == 0, res.stderr
    out = pd.read_csv(tmp_path / "known_target_priors.csv")
    assert set(out["target_id"]) == {"P1", "P2"}


def test_stage3_known_target_prior_applies_aligned_known_target_weights(
    tmp_path: Path,
) -> None:
    compound = write_compound_json(tmp_path / "compound.json", "CCO")
    pd.DataFrame([
        {
            "case_id": "weighted_exact",
            "smiles": "CCO",
            "known_targets": "P1;P2",
            "known_target_weights": "0.25;0.75",
            "source_weight": "1",
        }
    ]).to_csv(tmp_path / "panel.csv", index=False)

    res = run_prior(tmp_path, compound, tmp_path / "panel.csv")

    assert res.returncode == 0, res.stderr
    out = pd.read_csv(tmp_path / "known_target_priors.csv").set_index("target_id")
    assert out.loc["P1", "prior_score"] == 0.25
    assert out.loc["P2", "prior_score"] == 0.75


def test_stage3_known_target_prior_rejects_misaligned_known_target_weights(
    tmp_path: Path,
) -> None:
    compound = write_compound_json(tmp_path / "compound.json", "CCO")
    pd.DataFrame([
        {
            "case_id": "bad_weights",
            "smiles": "CCO",
            "known_targets": "P1;P2",
            "known_target_weights": "0.25",
        }
    ]).to_csv(tmp_path / "panel.csv", index=False)

    res = run_prior(tmp_path, compound, tmp_path / "panel.csv")

    assert res.returncode != 0
    assert "known_target_weights' count must match known_targets count" in res.stderr


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        (
            "source_weight",
            "",
            "Panel column 'source_weight' contains blank values",
        ),
        (
            "known_target_weights",
            " ",
            "Panel column 'known_target_weights' contains blank values",
        ),
    ],
)
def test_stage3_known_target_prior_rejects_blank_optional_weights(
    tmp_path: Path,
    column: str,
    value: str,
    message: str,
) -> None:
    compound = write_compound_json(tmp_path / "compound.json", "CCO")
    row = {
        "case_id": "blank_weight",
        "smiles": "CCO",
        "known_targets": "P1;P2",
        "source_weight": "1",
        "known_target_weights": "0.5;0.5",
    }
    row[column] = value
    pd.DataFrame([row]).to_csv(tmp_path / "panel.csv", index=False)

    res = run_prior(tmp_path, compound, tmp_path / "panel.csv")

    assert res.returncode != 0
    assert message in res.stderr


def test_stage3_known_target_prior_bayesian_or_combination_for_multiple_evidence_rows(
    tmp_path: Path,
) -> None:
    compound = write_compound_json(tmp_path / "compound.json", "CCOC")
    pd.DataFrame([
        {
            "case_id": "case_1",
            "smiles": "CCOC",
            "known_targets": "P1",
            "source_weight": "0.5",
        },
        {
            "case_id": "case_2",
            "smiles": "CCOC",
            "known_targets": "P1",
            "source_weight": "0.5",
        },
    ]).to_csv(tmp_path / "panel.csv", index=False)

    res = run_prior(
        tmp_path,
        compound,
        tmp_path / "panel.csv",
        args=["--prior-combination", "bayesian_or"],
    )

    assert res.returncode == 0, res.stderr
    out = pd.read_csv(tmp_path / "known_target_priors.csv")
    assert out.shape[0] == 1
    assert out.loc[0, "target_id"] == "P1"
    assert abs(out.loc[0, "prior_score"] - 0.75) < 1e-9


def test_stage3_known_target_prior_sum_capped_combination_for_multiple_evidence_rows(
    tmp_path: Path,
) -> None:
    compound = write_compound_json(tmp_path / "compound.json", "CCOC")
    pd.DataFrame([
        {
            "case_id": "case_1",
            "smiles": "CCOC",
            "known_targets": "P1",
            "source_weight": "0.6",
        },
        {
            "case_id": "case_2",
            "smiles": "CCOC",
            "known_targets": "P1",
            "source_weight": "0.6",
        },
    ]).to_csv(tmp_path / "panel.csv", index=False)

    res = run_prior(
        tmp_path,
        compound,
        tmp_path / "panel.csv",
        args=["--prior-combination", "sum_capped"],
    )

    assert res.returncode == 0, res.stderr
    out = pd.read_csv(tmp_path / "known_target_priors.csv")
    assert out.shape[0] == 1
    assert out.loc[0, "target_id"] == "P1"
    assert out.loc[0, "prior_score"] == 1.0


def test_stage3_known_target_prior_rejects_duplicate_targets_within_case(
    tmp_path: Path,
) -> None:
    compound = write_compound_json(tmp_path / "compound.json", "CCO")
    pd.DataFrame([
        {
            "case_id": "duplicate_target",
            "smiles": "CCO",
            "known_targets": "P1;P1",
            "known_target_weights": "0.25;0.75",
        }
    ]).to_csv(tmp_path / "panel.csv", index=False)

    res = run_prior(tmp_path, compound, tmp_path / "panel.csv")

    assert res.returncode != 0
    assert "known_targets contains duplicate target(s) P1" in res.stderr
