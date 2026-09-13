"""Unit tests for the REINVENT-funnel sizing (§10.3)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage5_6_mini_validate import FunnelConfig, funnel_sizes  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CFG = FunnelConfig(keep_admet=500, keep_route_proxy=300, keep_drug=250, keep_boltz=30)


def write_reference_parquets(tmp_path: Path) -> tuple[Path, Path]:
    cosing = tmp_path / "cosing.parquet"
    drugs = tmp_path / "drugs.parquet"
    pd.DataFrame([{"inci_name": "Reference", "smiles": "CCN"}]).to_parquet(cosing)
    pd.DataFrame([{"drug_id": "DRUG1", "ecfp4": [0] * 32}]).to_parquet(drugs)
    return cosing, drugs


def packed_ecfp4(smiles: str) -> list[int]:
    from rdkit import Chem
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    import numpy as np

    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = generator.GetFingerprint(mol)
    bits = np.zeros((2048,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, bits)
    return [int(word) for word in np.frombuffer(np.packbits(bits).tobytes(), dtype=np.uint64)]


def run_mini_validate(
    tmp_path: Path,
    *,
    in_smi: Path,
    cosing: Path | None = None,
    drugs: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    if cosing is None or drugs is None:
        default_cosing, default_drugs = write_reference_parquets(tmp_path)
        cosing = cosing or default_cosing
        drugs = drugs or default_drugs
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage5_6_mini_validate.py"),
            "--in-smi",
            str(in_smi),
            "--cosing-parquet",
            str(cosing),
            "--drugs-parquet",
            str(drugs),
            "--out-sdf",
            str(tmp_path / "top30.sdf"),
            "--out-csv",
            str(tmp_path / "top30.csv"),
            "--out-lineage",
            str(tmp_path / "lineage.json"),
            "--out-funnel",
            str(tmp_path / "funnel.tsv"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def write_stale_outputs(tmp_path: Path) -> None:
    for name in ("top30.sdf", "top30.csv", "lineage.json", "funnel.tsv"):
        (tmp_path / name).write_text("stale\n")


def assert_no_outputs(tmp_path: Path) -> None:
    for name in ("top30.sdf", "top30.csv", "lineage.json", "funnel.tsv"):
        assert not (tmp_path / name).exists()


def test_funnel_at_5000_yields_canonical_sizes() -> None:
    sizes = funnel_sizes(5000, CFG)
    assert sizes == {"raw": 5000, "admet": 500, "route_proxy": 300,
                     "drug": 250, "boltz": 30}


def test_funnel_monotonically_nonincreasing() -> None:
    sizes = funnel_sizes(5000, CFG)
    ordered = [sizes["raw"], sizes["admet"], sizes["route_proxy"],
               sizes["drug"], sizes["boltz"]]
    assert ordered == sorted(ordered, reverse=True)


def test_funnel_clamps_when_raw_smaller_than_caps() -> None:
    sizes = funnel_sizes(50, CFG)
    # ADMET cap=500 but raw=50 → 50
    # Routeability proxy cap=300 → 50
    # Drug cap=250 → 50
    # Boltz cap=30 → 30
    assert sizes == {"raw": 50, "admet": 50, "route_proxy": 50,
                     "drug": 50, "boltz": 30}


def test_funnel_zero_raw() -> None:
    sizes = funnel_sizes(0, CFG)
    assert sizes == {"raw": 0, "admet": 0, "route_proxy": 0,
                     "drug": 0, "boltz": 0}


def test_custom_caps_respected() -> None:
    cfg = FunnelConfig(keep_admet=200, keep_route_proxy=100, keep_drug=50, keep_boltz=10)
    sizes = funnel_sizes(1000, cfg)
    assert sizes == {"raw": 1000, "admet": 200, "route_proxy": 100,
                     "drug": 50, "boltz": 10}


def test_mini_validate_fails_when_reinvent_input_missing(tmp_path: Path) -> None:
    write_stale_outputs(tmp_path)

    res = run_mini_validate(tmp_path, in_smi=tmp_path / "missing.smi")

    assert res.returncode != 0
    assert "REINVENT analog input is required" in res.stderr
    assert_no_outputs(tmp_path)


def test_mini_validate_fails_when_reinvent_input_empty(tmp_path: Path) -> None:
    in_smi = tmp_path / "generated.smi"
    in_smi.write_text("")
    write_stale_outputs(tmp_path)

    res = run_mini_validate(tmp_path, in_smi=in_smi)

    assert res.returncode != 0
    assert "REINVENT analog input is empty" in res.stderr
    assert_no_outputs(tmp_path)


def test_mini_validate_fails_when_all_smiles_invalid(tmp_path: Path) -> None:
    in_smi = tmp_path / "generated.smi"
    in_smi.write_text("not_a_smiles\n")
    write_stale_outputs(tmp_path)

    res = run_mini_validate(tmp_path, in_smi=in_smi)

    assert res.returncode != 0
    assert "REINVENT analog input contains invalid SMILES" in res.stderr
    assert_no_outputs(tmp_path)


def test_mini_validate_fails_when_any_smiles_invalid(tmp_path: Path) -> None:
    in_smi = tmp_path / "generated.smi"
    in_smi.write_text("CCO analog1\nnot_a_smiles analog2\n")
    write_stale_outputs(tmp_path)

    res = run_mini_validate(tmp_path, in_smi=in_smi)

    assert res.returncode != 0
    assert "REINVENT analog input contains invalid SMILES" in res.stderr
    assert "row index(es) 1" in res.stderr
    assert_no_outputs(tmp_path)


def test_mini_validate_fails_when_reinvent_input_has_duplicate_smiles(
    tmp_path: Path,
) -> None:
    in_smi = tmp_path / "generated.smi"
    in_smi.write_text("CCO analog1\nCCO analog2\n")
    write_stale_outputs(tmp_path)

    res = run_mini_validate(tmp_path, in_smi=in_smi)

    assert res.returncode != 0
    assert "REINVENT analog input contains duplicate SMILES: CCO" in res.stderr
    assert_no_outputs(tmp_path)


def test_mini_validate_fails_when_reinvent_input_has_canonical_duplicate_smiles(
    tmp_path: Path,
) -> None:
    in_smi = tmp_path / "generated.smi"
    in_smi.write_text("CCO analog1\nOCC analog2\n")
    write_stale_outputs(tmp_path)

    res = run_mini_validate(tmp_path, in_smi=in_smi)

    assert res.returncode != 0
    assert "REINVENT analog input contains duplicate canonical SMILES: CCO" in res.stderr
    assert_no_outputs(tmp_path)


def test_mini_validate_fails_when_drug_reference_empty(tmp_path: Path) -> None:
    in_smi = tmp_path / "generated.smi"
    in_smi.write_text("CCO analog1\n")
    cosing, drugs = write_reference_parquets(tmp_path)
    pd.DataFrame(columns=["ecfp4"]).to_parquet(drugs)
    write_stale_outputs(tmp_path)

    res = run_mini_validate(tmp_path, in_smi=in_smi, cosing=cosing, drugs=drugs)

    assert res.returncode != 0
    assert "Drug reference parquet contains no rows" in res.stderr
    assert_no_outputs(tmp_path)


def test_mini_validate_fails_when_drug_fingerprint_malformed(tmp_path: Path) -> None:
    in_smi = tmp_path / "generated.smi"
    in_smi.write_text("CCO analog1\n")
    cosing, drugs = write_reference_parquets(tmp_path)
    pd.DataFrame([{"drug_id": "DRUG1", "ecfp4": [0, 1]}]).to_parquet(drugs)
    write_stale_outputs(tmp_path)

    res = run_mini_validate(tmp_path, in_smi=in_smi, cosing=cosing, drugs=drugs)

    assert res.returncode != 0
    assert (
        "Drug reference parquet column 'ecfp4' must contain 32 packed uint64 integers"
        in res.stderr
    )
    assert_no_outputs(tmp_path)


def test_mini_validate_fails_when_drug_fingerprint_is_boolean(tmp_path: Path) -> None:
    in_smi = tmp_path / "generated.smi"
    in_smi.write_text("CCO analog1\n")
    cosing, drugs = write_reference_parquets(tmp_path)
    pd.DataFrame([{"drug_id": "DRUG1", "ecfp4": [True] * 32}]).to_parquet(drugs)
    write_stale_outputs(tmp_path)

    res = run_mini_validate(tmp_path, in_smi=in_smi, cosing=cosing, drugs=drugs)

    assert res.returncode != 0
    assert (
        "Drug reference parquet column 'ecfp4' must contain 32 packed uint64 integers"
        in res.stderr
    )
    assert_no_outputs(tmp_path)


def test_mini_validate_accepts_decimal_string_uint64_fingerprint_words(
    tmp_path: Path,
) -> None:
    in_smi = tmp_path / "generated.smi"
    in_smi.write_text("CCO analog1\n")
    cosing, drugs = write_reference_parquets(tmp_path)
    pd.DataFrame(
        [{"drug_id": "DRUG1", "ecfp4": [str(2**63)] + ["0"] * 31}]
    ).to_parquet(drugs)

    res = run_mini_validate(tmp_path, in_smi=in_smi, cosing=cosing, drugs=drugs)

    assert res.returncode == 0, res.stderr
    assert (tmp_path / "top30.csv").exists()


def test_mini_validate_fails_when_cosing_reference_missing_smiles(
    tmp_path: Path,
) -> None:
    in_smi = tmp_path / "generated.smi"
    in_smi.write_text("CCO analog1\n")
    cosing, drugs = write_reference_parquets(tmp_path)
    pd.DataFrame([{"inci_name": "Reference"}]).to_parquet(cosing)
    write_stale_outputs(tmp_path)

    res = run_mini_validate(tmp_path, in_smi=in_smi, cosing=cosing, drugs=drugs)

    assert res.returncode != 0
    assert "CosIng reference parquet missing required columns ['smiles']" in res.stderr
    assert_no_outputs(tmp_path)


def test_mini_validate_fails_when_cosing_reference_smiles_invalid(
    tmp_path: Path,
) -> None:
    in_smi = tmp_path / "generated.smi"
    in_smi.write_text("CCO analog1\n")
    cosing, drugs = write_reference_parquets(tmp_path)
    pd.DataFrame([{"inci_name": "Reference", "smiles": "not_a_smiles"}]).to_parquet(
        cosing
    )
    write_stale_outputs(tmp_path)

    res = run_mini_validate(tmp_path, in_smi=in_smi, cosing=cosing, drugs=drugs)

    assert res.returncode != 0
    assert "CosIng reference parquet column 'smiles' contains invalid SMILES" in res.stderr
    assert_no_outputs(tmp_path)


def test_mini_validate_writes_outputs_for_valid_analog(tmp_path: Path) -> None:
    in_smi = tmp_path / "generated.smi"
    in_smi.write_text("CCO analog1\n")

    res = run_mini_validate(tmp_path, in_smi=in_smi)

    assert res.returncode == 0, res.stderr
    out = pd.read_csv(tmp_path / "top30.csv")
    assert out["smiles"].tolist() == ["CCO"]
    assert {"cosing_tanimoto", "drug_tanimoto"} <= set(out.columns)
    assert (tmp_path / "top30.sdf").stat().st_size > 0
    assert '"final_count": 1' in (tmp_path / "lineage.json").read_text()
    assert '"cosing_reference_count": 1' in (tmp_path / "lineage.json").read_text()
    assert '"drug_reference_count": 1' in (tmp_path / "lineage.json").read_text()
    assert (tmp_path / "funnel.tsv").read_text().endswith("boltz\t1\n")
    assert "route_proxy\t1\n" in (tmp_path / "funnel.tsv").read_text()
    assert "rank_slice_proxy_not_aizynth" in (tmp_path / "lineage.json").read_text()


def test_mini_validate_records_actual_drug_survivor_count(tmp_path: Path) -> None:
    in_smi = tmp_path / "generated.smi"
    in_smi.write_text("CCO analog1\nCCN analog2\n")
    cosing, drugs = write_reference_parquets(tmp_path)
    pd.DataFrame([{"drug_id": "DRUG1", "ecfp4": packed_ecfp4("CCO")}]).to_parquet(
        drugs
    )

    res = run_mini_validate(tmp_path, in_smi=in_smi, cosing=cosing, drugs=drugs)

    assert res.returncode == 0, res.stderr
    out = pd.read_csv(tmp_path / "top30.csv")
    assert out["smiles"].tolist() == ["CCN"]
    funnel = (tmp_path / "funnel.tsv").read_text()
    assert "drug\t1\n" in funnel
    assert "boltz\t1\n" in funnel
