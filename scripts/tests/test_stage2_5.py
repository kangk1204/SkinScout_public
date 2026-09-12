"""Unit tests for Stage 2.5 — cosmetic annotation + drug-avoidance decision."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage2_5_cosing_match import classify              # noqa: E402
from stage2_5_decision import decide, degraded_references, framing  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def write_input_sdf(path: Path) -> None:
    mol = Chem.MolFromSmiles("CCO")
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()


def run_script(script: str, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *args],
        capture_output=True,
        text=True,
        check=False,
    )


# --- CosIng tier classification ---------------------------------------------


def test_cosing_classify_exact_via_inchi() -> None:
    assert classify(0.0, True, 0.85, 0.65) == "EXACT"


def test_cosing_classify_exact_via_tanimoto_one() -> None:
    assert classify(1.0, False, 0.85, 0.65) == "EXACT"


def test_cosing_classify_similar() -> None:
    assert classify(0.90, False, 0.85, 0.65) == "SIMILAR"


def test_cosing_classify_analog() -> None:
    assert classify(0.70, False, 0.85, 0.65) == "ANALOG"


def test_cosing_classify_new() -> None:
    assert classify(0.40, False, 0.85, 0.65) == "NEW"


# --- Drug-policy decision matrix --------------------------------------------


def _drug(*tiers: str) -> dict:
    return {"warnings": [{"tier": t} for t in tiers]}


@pytest.fixture
def cosing_new() -> dict:
    return {"level": "NEW", "inci": None}


def test_decide_strict_halts_on_strict_warning(cosing_new: dict) -> None:
    assert decide(cosing_new, _drug("STRICT_WARNING"), "strict") == "HALT"


def test_decide_strict_halts_on_scaffold_match(cosing_new: dict) -> None:
    assert decide(cosing_new, _drug("SCAFFOLD_MATCH"), "strict") == "HALT"


def test_decide_strict_downweights_soft(cosing_new: dict) -> None:
    assert decide(cosing_new, _drug("SOFT_WARNING"), "strict") == "DOWNWEIGHT"


def test_decide_strict_proceeds_when_clean(cosing_new: dict) -> None:
    assert decide(cosing_new, _drug(), "strict") == "PROCEED"


def test_decide_moderate_downweights_strict(cosing_new: dict) -> None:
    assert decide(cosing_new, _drug("STRICT_WARNING"), "moderate") == "DOWNWEIGHT"


def test_decide_moderate_proceeds_on_soft(cosing_new: dict) -> None:
    assert decide(cosing_new, _drug("SOFT_WARNING"), "moderate") == "PROCEED"


def test_decide_lenient_always_proceeds(cosing_new: dict) -> None:
    assert decide(cosing_new, _drug("STRICT_WARNING", "SCAFFOLD_MATCH"), "lenient") == "PROCEED"


def test_decide_invalid_policy_raises(cosing_new: dict) -> None:
    with pytest.raises(ValueError):
        decide(cosing_new, _drug(), "ultra-mega-strict")


def test_degraded_reference_status_is_reported() -> None:
    degraded = degraded_references(
        {"reference_status": "missing_degraded"},
        {"reference_status": "ok"},
    )

    assert degraded == ["CosIng reference_status=missing_degraded"]


# --- Framing strings --------------------------------------------------------


@pytest.mark.parametrize(
    "level,expected_phrase",
    [
        ("EXACT", "existing cosmetic ingredient"),
        ("SIMILAR", "very similar"),
        ("ANALOG", "analog"),
        ("NEW", "novel"),
    ],
)
def test_framing_strings(level: str, expected_phrase: str) -> None:
    f = framing({"level": level, "inci": "Niacinamide"})
    assert expected_phrase in f


# --- Reference availability gates -------------------------------------------


def test_cosing_missing_reference_fails_without_degraded_flag(tmp_path: Path) -> None:
    sdf = tmp_path / "ligand.sdf"
    write_input_sdf(sdf)
    (tmp_path / "cosing.json").write_text("stale\n")

    res = run_script(
        "stage2_5_cosing_match.py",
        [
            "--in-sdf",
            str(sdf),
            "--cosing-parquet",
            str(tmp_path / "missing.parquet"),
            "--out-json",
            str(tmp_path / "cosing.json"),
        ],
    )

    assert res.returncode != 0
    assert "CosIng parquet is required" in res.stderr
    assert not (tmp_path / "cosing.json").exists()


def test_cosing_missing_reference_degraded_mode_is_explicit(tmp_path: Path) -> None:
    sdf = tmp_path / "ligand.sdf"
    write_input_sdf(sdf)

    res = run_script(
        "stage2_5_cosing_match.py",
        [
            "--in-sdf",
            str(sdf),
            "--cosing-parquet",
            str(tmp_path / "missing.parquet"),
            "--out-json",
            str(tmp_path / "cosing.json"),
            "--allow-missing-reference",
        ],
    )

    assert res.returncode == 0, res.stderr
    payload = json.loads((tmp_path / "cosing.json").read_text())
    assert payload["level"] == "NEW"
    assert payload["reference_status"] == "missing_degraded"


def test_cosing_empty_reference_fails_without_degraded_flag(tmp_path: Path) -> None:
    sdf = tmp_path / "ligand.sdf"
    write_input_sdf(sdf)
    cosing = tmp_path / "cosing.parquet"
    pd.DataFrame(columns=["inchikey", "ecfp4", "inci_name", "functions"]).to_parquet(cosing)
    (tmp_path / "cosing.json").write_text("stale\n")

    res = run_script(
        "stage2_5_cosing_match.py",
        [
            "--in-sdf",
            str(sdf),
            "--cosing-parquet",
            str(cosing),
            "--out-json",
            str(tmp_path / "cosing.json"),
        ],
    )

    assert res.returncode != 0
    assert "CosIng parquet is empty" in res.stderr
    assert not (tmp_path / "cosing.json").exists()


def test_cosing_malformed_fingerprint_fails_without_json(tmp_path: Path) -> None:
    sdf = tmp_path / "ligand.sdf"
    write_input_sdf(sdf)
    cosing = tmp_path / "cosing.parquet"
    pd.DataFrame(
        [
            {
                "inchikey": "TEST",
                "ecfp4": [0, 1],
                "inci_name": "Bad Reference",
                "functions": "Skin Conditioning",
            }
        ]
    ).to_parquet(cosing)
    (tmp_path / "cosing.json").write_text("stale\n")

    res = run_script(
        "stage2_5_cosing_match.py",
        [
            "--in-sdf",
            str(sdf),
            "--cosing-parquet",
            str(cosing),
            "--out-json",
            str(tmp_path / "cosing.json"),
        ],
    )

    assert res.returncode != 0
    assert (
        "CosIng parquet column 'ecfp4' must contain 32 packed uint64 integers"
        in res.stderr
    )
    assert not (tmp_path / "cosing.json").exists()


def test_cosing_decimal_string_uint64_fingerprint_words_are_accepted(
    tmp_path: Path,
) -> None:
    sdf = tmp_path / "ligand.sdf"
    write_input_sdf(sdf)
    cosing = tmp_path / "cosing.parquet"
    pd.DataFrame(
        [
            {
                "inchikey": "TEST",
                "ecfp4": [str(2**63)] + ["0"] * 31,
                "inci_name": "String Fingerprint Reference",
                "functions": "Skin Conditioning",
            }
        ]
    ).to_parquet(cosing)

    res = run_script(
        "stage2_5_cosing_match.py",
        [
            "--in-sdf",
            str(sdf),
            "--cosing-parquet",
            str(cosing),
            "--out-json",
            str(tmp_path / "cosing.json"),
        ],
    )

    assert res.returncode == 0, res.stderr
    assert (tmp_path / "cosing.json").exists()


@pytest.mark.parametrize(
    ("column", "message"),
    [
        ("inchikey", "CosIng parquet column 'inchikey' contains blank values"),
        ("inci_name", "CosIng parquet column 'inci_name' contains blank values"),
    ],
)
def test_cosing_blank_required_text_fields_fail_without_json(
    tmp_path: Path,
    column: str,
    message: str,
) -> None:
    sdf = tmp_path / "ligand.sdf"
    write_input_sdf(sdf)
    cosing = tmp_path / "cosing.parquet"
    row = {
        "inchikey": "TEST",
        "ecfp4": [0] * 32,
        "inci_name": "Reference",
        "functions": "Skin Conditioning",
    }
    row[column] = " "
    pd.DataFrame([row]).to_parquet(cosing)
    (tmp_path / "cosing.json").write_text("stale\n")

    res = run_script(
        "stage2_5_cosing_match.py",
        [
            "--in-sdf",
            str(sdf),
            "--cosing-parquet",
            str(cosing),
            "--out-json",
            str(tmp_path / "cosing.json"),
        ],
    )

    assert res.returncode != 0
    assert message in res.stderr
    assert not (tmp_path / "cosing.json").exists()


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        (
            "--similar-threshold",
            "-0.1",
            "--similar-threshold must be a finite value in [0, 1]: -0.1",
        ),
        (
            "--similar-threshold",
            "nan",
            "--similar-threshold must be a finite value in [0, 1]: nan",
        ),
        (
            "--analog-threshold",
            "1.1",
            "--analog-threshold must be a finite value in [0, 1]: 1.1",
        ),
        (
            "--analog-threshold",
            "0.9",
            "--similar-threshold must be >= --analog-threshold: 0.85 < 0.9",
        ),
    ],
)
def test_cosing_invalid_thresholds_fail_without_json(
    tmp_path: Path,
    flag: str,
    value: str,
    message: str,
) -> None:
    sdf = tmp_path / "ligand.sdf"
    write_input_sdf(sdf)
    out = tmp_path / "cosing.json"
    out.write_text("stale\n")

    res = run_script(
        "stage2_5_cosing_match.py",
        [
            "--in-sdf",
            str(sdf),
            "--cosing-parquet",
            str(tmp_path / "missing.parquet"),
            "--out-json",
            str(out),
            flag,
            value,
            "--allow-missing-reference",
        ],
    )

    assert res.returncode != 0
    assert message in res.stderr
    assert not out.exists()


def test_drug_avoidance_missing_reference_fails_without_degraded_flag(tmp_path: Path) -> None:
    sdf = tmp_path / "ligand.sdf"
    write_input_sdf(sdf)
    (tmp_path / "drug.json").write_text("stale\n")

    res = run_script(
        "stage2_5_drug_avoidance.py",
        [
            "--in-sdf",
            str(sdf),
            "--drugs-parquet",
            str(tmp_path / "missing_drugs.parquet"),
            "--scaffolds-parquet",
            str(tmp_path / "missing_scaffolds.parquet"),
            "--out-json",
            str(tmp_path / "drug.json"),
        ],
    )

    assert res.returncode != 0
    assert "reference file(s) required" in res.stderr
    assert not (tmp_path / "drug.json").exists()


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        (
            "--strict-threshold",
            "-0.1",
            "--strict-threshold must be a finite value in [0, 1]: -0.1",
        ),
        (
            "--strict-threshold",
            "nan",
            "--strict-threshold must be a finite value in [0, 1]: nan",
        ),
        (
            "--soft-threshold",
            "1.1",
            "--soft-threshold must be a finite value in [0, 1]: 1.1",
        ),
        (
            "--soft-threshold",
            "0.9",
            "--strict-threshold must be >= --soft-threshold: 0.85 < 0.9",
        ),
    ],
)
def test_drug_avoidance_invalid_thresholds_fail_without_json(
    tmp_path: Path,
    flag: str,
    value: str,
    message: str,
) -> None:
    sdf = tmp_path / "ligand.sdf"
    write_input_sdf(sdf)
    out = tmp_path / "drug.json"
    out.write_text("stale\n")

    res = run_script(
        "stage2_5_drug_avoidance.py",
        [
            "--in-sdf",
            str(sdf),
            "--drugs-parquet",
            str(tmp_path / "missing_drugs.parquet"),
            "--scaffolds-parquet",
            str(tmp_path / "missing_scaffolds.parquet"),
            "--out-json",
            str(out),
            flag,
            value,
            "--allow-missing-reference",
        ],
    )

    assert res.returncode != 0
    assert message in res.stderr
    assert not out.exists()


def test_drug_avoidance_missing_reference_degraded_mode_is_explicit(tmp_path: Path) -> None:
    sdf = tmp_path / "ligand.sdf"
    write_input_sdf(sdf)

    res = run_script(
        "stage2_5_drug_avoidance.py",
        [
            "--in-sdf",
            str(sdf),
            "--drugs-parquet",
            str(tmp_path / "missing_drugs.parquet"),
            "--scaffolds-parquet",
            str(tmp_path / "missing_scaffolds.parquet"),
            "--out-json",
            str(tmp_path / "drug.json"),
            "--allow-missing-reference",
        ],
    )

    assert res.returncode == 0, res.stderr
    payload = json.loads((tmp_path / "drug.json").read_text())
    assert payload["warnings"] == []
    assert payload["reference_status"] == "missing_degraded"


def test_drug_avoidance_empty_reference_fails_without_degraded_flag(tmp_path: Path) -> None:
    sdf = tmp_path / "ligand.sdf"
    write_input_sdf(sdf)
    drugs = tmp_path / "drugs.parquet"
    scaffolds = tmp_path / "scaffolds.parquet"
    pd.DataFrame(columns=["ecfp4", "drug_id", "name"]).to_parquet(drugs)
    pd.DataFrame(columns=["scaffold_smiles"]).to_parquet(scaffolds)
    (tmp_path / "drug.json").write_text("stale\n")

    res = run_script(
        "stage2_5_drug_avoidance.py",
        [
            "--in-sdf",
            str(sdf),
            "--drugs-parquet",
            str(drugs),
            "--scaffolds-parquet",
            str(scaffolds),
            "--out-json",
            str(tmp_path / "drug.json"),
        ],
    )

    assert res.returncode != 0
    assert "Approved-drug parquet is empty" in res.stderr
    assert not (tmp_path / "drug.json").exists()


def test_drug_avoidance_malformed_fingerprint_fails_without_json(
    tmp_path: Path,
) -> None:
    sdf = tmp_path / "ligand.sdf"
    write_input_sdf(sdf)
    drugs = tmp_path / "drugs.parquet"
    scaffolds = tmp_path / "scaffolds.parquet"
    pd.DataFrame([{"ecfp4": [0, 1], "drug_id": "DRUG1", "name": "Drug"}]).to_parquet(
        drugs
    )
    pd.DataFrame([{"scaffold_smiles": "CC"}]).to_parquet(scaffolds)
    (tmp_path / "drug.json").write_text("stale\n")

    res = run_script(
        "stage2_5_drug_avoidance.py",
        [
            "--in-sdf",
            str(sdf),
            "--drugs-parquet",
            str(drugs),
            "--scaffolds-parquet",
            str(scaffolds),
            "--out-json",
            str(tmp_path / "drug.json"),
        ],
    )

    assert res.returncode != 0
    assert (
        "Approved-drug parquet column 'ecfp4' must contain 32 packed uint64 integers"
        in res.stderr
    )
    assert not (tmp_path / "drug.json").exists()


def test_drug_avoidance_decimal_string_uint64_fingerprint_words_are_accepted(
    tmp_path: Path,
) -> None:
    sdf = tmp_path / "ligand.sdf"
    write_input_sdf(sdf)
    drugs = tmp_path / "drugs.parquet"
    scaffolds = tmp_path / "scaffolds.parquet"
    pd.DataFrame(
        [
            {
                "ecfp4": [str(2**63)] + ["0"] * 31,
                "drug_id": "DRUG1",
                "name": "Drug",
            }
        ]
    ).to_parquet(drugs)
    pd.DataFrame([{"scaffold_smiles": "CC"}]).to_parquet(scaffolds)

    res = run_script(
        "stage2_5_drug_avoidance.py",
        [
            "--in-sdf",
            str(sdf),
            "--drugs-parquet",
            str(drugs),
            "--scaffolds-parquet",
            str(scaffolds),
            "--out-json",
            str(tmp_path / "drug.json"),
        ],
    )

    assert res.returncode == 0, res.stderr
    assert (tmp_path / "drug.json").exists()


@pytest.mark.parametrize(
    ("column", "message"),
    [
        ("drug_id", "Approved-drug parquet column 'drug_id' contains blank values"),
        ("name", "Approved-drug parquet column 'name' contains blank values"),
    ],
)
def test_drug_avoidance_blank_required_text_fields_fail_without_json(
    tmp_path: Path,
    column: str,
    message: str,
) -> None:
    sdf = tmp_path / "ligand.sdf"
    write_input_sdf(sdf)
    drugs = tmp_path / "drugs.parquet"
    scaffolds = tmp_path / "scaffolds.parquet"
    row = {"ecfp4": [0] * 32, "drug_id": "DRUG1", "name": "Drug"}
    row[column] = " "
    pd.DataFrame([row]).to_parquet(drugs)
    pd.DataFrame([{"scaffold_smiles": "CC"}]).to_parquet(scaffolds)
    (tmp_path / "drug.json").write_text("stale\n")

    res = run_script(
        "stage2_5_drug_avoidance.py",
        [
            "--in-sdf",
            str(sdf),
            "--drugs-parquet",
            str(drugs),
            "--scaffolds-parquet",
            str(scaffolds),
            "--out-json",
            str(tmp_path / "drug.json"),
        ],
    )

    assert res.returncode != 0
    assert message in res.stderr
    assert not (tmp_path / "drug.json").exists()


def test_drug_avoidance_invalid_scaffold_smiles_fails_without_json(
    tmp_path: Path,
) -> None:
    sdf = tmp_path / "ligand.sdf"
    write_input_sdf(sdf)
    drugs = tmp_path / "drugs.parquet"
    scaffolds = tmp_path / "scaffolds.parquet"
    pd.DataFrame([{"ecfp4": [0] * 32, "drug_id": "DRUG1", "name": "Drug"}]).to_parquet(
        drugs
    )
    pd.DataFrame([{"scaffold_smiles": "not_a_smiles"}]).to_parquet(scaffolds)
    (tmp_path / "drug.json").write_text("stale\n")

    res = run_script(
        "stage2_5_drug_avoidance.py",
        [
            "--in-sdf",
            str(sdf),
            "--drugs-parquet",
            str(drugs),
            "--scaffolds-parquet",
            str(scaffolds),
            "--out-json",
            str(tmp_path / "drug.json"),
        ],
    )

    assert res.returncode != 0
    assert "scaffold_smiles' contains invalid SMILES" in res.stderr
    assert not (tmp_path / "drug.json").exists()


def test_drug_avoidance_blank_acyclic_scaffold_does_not_match_every_acyclic_ligand(
    tmp_path: Path,
) -> None:
    sdf = tmp_path / "ligand.sdf"
    write_input_sdf(sdf)
    drugs = tmp_path / "drugs.parquet"
    scaffolds = tmp_path / "scaffolds.parquet"
    pd.DataFrame([{"ecfp4": [0] * 32, "drug_id": "DRUG1", "name": "Drug"}]).to_parquet(
        drugs
    )
    pd.DataFrame([{"scaffold_smiles": ""}]).to_parquet(scaffolds)

    res = run_script(
        "stage2_5_drug_avoidance.py",
        [
            "--in-sdf",
            str(sdf),
            "--drugs-parquet",
            str(drugs),
            "--scaffolds-parquet",
            str(scaffolds),
            "--out-json",
            str(tmp_path / "drug.json"),
        ],
    )

    assert res.returncode == 0, res.stderr
    payload = json.loads((tmp_path / "drug.json").read_text())
    assert all(warning["tier"] != "SCAFFOLD_MATCH" for warning in payload["warnings"])


def test_decision_fails_on_degraded_references_by_default(tmp_path: Path) -> None:
    cosing = tmp_path / "cosing.json"
    drug = tmp_path / "drug.json"
    out = tmp_path / "decision.txt"
    cosing.write_text(json.dumps({"level": "NEW", "reference_status": "missing_degraded"}))
    drug.write_text(json.dumps({"warnings": [], "reference_status": "ok"}))
    out.write_text("stale\n")

    res = run_script(
        "stage2_5_decision.py",
        [
            "--cosing-json",
            str(cosing),
            "--drug-json",
            str(drug),
            "--out-decision",
            str(out),
        ],
    )

    assert res.returncode != 0
    assert "decision requires full references" in res.stderr
    assert not out.exists()


def test_decision_allows_degraded_references_only_for_diagnostics(tmp_path: Path) -> None:
    cosing = tmp_path / "cosing.json"
    drug = tmp_path / "drug.json"
    out = tmp_path / "decision.txt"
    cosing.write_text(json.dumps({"level": "NEW", "reference_status": "ok"}))
    drug.write_text(json.dumps({"warnings": [], "reference_status": "empty_degraded"}))

    res = run_script(
        "stage2_5_decision.py",
        [
            "--cosing-json",
            str(cosing),
            "--drug-json",
            str(drug),
            "--out-decision",
            str(out),
            "--allow-degraded-references",
        ],
    )

    assert res.returncode == 0, res.stderr
    assert out.read_text().startswith("PROCEED")


def test_decision_fails_cleanly_when_drug_warnings_missing_list(
    tmp_path: Path,
) -> None:
    cosing = tmp_path / "cosing.json"
    drug = tmp_path / "drug.json"
    out = tmp_path / "decision.txt"
    cosing.write_text(json.dumps({"level": "NEW", "reference_status": "ok"}))
    drug.write_text(json.dumps({"reference_status": "ok"}))

    res = run_script(
        "stage2_5_decision.py",
        [
            "--cosing-json",
            str(cosing),
            "--drug-json",
            str(drug),
            "--out-decision",
            str(out),
        ],
    )

    assert res.returncode != 0
    assert "warnings field 'warnings' must be a list" in res.stderr
    assert "Traceback" not in res.stderr
    assert not out.exists()


def _drug_avoidance_inputs(tmp_path: Path) -> list[str]:
    sdf = tmp_path / "ligand.sdf"
    write_input_sdf(sdf)
    drugs = tmp_path / "drugs.parquet"
    scaffolds = tmp_path / "scaffolds.parquet"
    pd.DataFrame(
        [{"ecfp4": ["0"] * 32, "drug_id": "DRUG1", "name": "Drug"}]
    ).to_parquet(drugs)
    pd.DataFrame([{"scaffold_smiles": "CC"}]).to_parquet(scaffolds)
    return [
        "--in-sdf", str(sdf),
        "--drugs-parquet", str(drugs),
        "--scaffolds-parquet", str(scaffolds),
    ]


def test_drug_avoidance_exits_cleanly_every_time(tmp_path: Path) -> None:
    """It used to abort after finishing, about one run in ten.

    RDKit's static teardown raised "terminate called without an active
    exception" and the process died with SIGABRT *after* the JSON had been
    written atomically and the summary logged. The stage failed with exit -6
    while its complete output sat on disk, which reads as a pipeline failure
    with no failed work behind it. Caught by a full-suite run, not by this
    file - a single invocation passes ~90% of the time.
    """
    base = _drug_avoidance_inputs(tmp_path)

    for attempt in range(12):
        out = tmp_path / f"drug{attempt}.json"
        result = run_script("stage2_5_drug_avoidance.py", [*base, "--out-json", str(out)])
        assert result.returncode == 0, (attempt, result.returncode, result.stderr)
        assert out.exists(), attempt


def test_drug_avoidance_still_reports_a_real_failure(tmp_path: Path) -> None:
    """The fix exits before C++ teardown, so it must not swallow exit codes."""
    base = _drug_avoidance_inputs(tmp_path)
    base[1] = str(tmp_path / "absent.sdf")

    result = run_script(
        "stage2_5_drug_avoidance.py", [*base, "--out-json", str(tmp_path / "drug.json")]
    )

    assert result.returncode != 0
    assert not (tmp_path / "drug.json").exists()
