from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build_bindingdb_known_target_panel.py"
_ = pytest.importorskip("rdkit")


def run_script(tmp_path: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(SCRIPT)] + args
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )


pd = pytest.importorskip("pandas")


def _write_bindingdb_tsv(path: Path, rows: list[str]) -> Path:
    header = (
        "Ligand SMILES\tLigand InChI Key\tTarget Source Organism According to Curator or DataSource\t"
        "UniProt (SwissProt) Primary ID of Target Chain 1\tUniProt (SwissProt) Recommended Name of Target Chain 1\t"
        "UniProt (SwissProt) Primary ID of Target Chain 2\tUniProt (SwissProt) Recommended Name of Target Chain 2\t"
        "Ki (nM)\tBindingDB Entry DOI\n"
    )
    path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_build_panel_collapses_to_one_case_row(tmp_path: Path) -> None:
    tsv = _write_bindingdb_tsv(
        tmp_path / "BindingDB_All.tsv",
        [
            "CCO\tBSYNRYMUTXBXSQ-UHFFFAOYSA-N\tHomo sapiens\tP11111\tTYR\tP22222\tDCT\t1\t10.1000/test1",
            "CCO\tBSYNRYMUTXBXSQ-UHFFFAOYSA-N\tHomo sapiens\tP33333\tAHR\t\t\t1000\t10.1000/test2",
        ],
    )
    out_csv = tmp_path / "out.csv"

    res = run_script(
        tmp_path,
        [
            "--bindingdb-tsv",
            str(tsv),
            "--out-csv",
            str(out_csv),
            "--max-targets-per-compound",
            "10",
        ],
    )

    assert res.returncode == 0, res.stderr
    out = pd.read_csv(out_csv)
    assert out.shape[0] == 1
    assert out.loc[0, "case_id"].startswith("BDB_")
    assert out.loc[0, "known_targets"] == "P11111;P22222;P33333"
    assert "known_target_weights" in out.columns


def test_build_panel_enforces_human_filter_by_default(tmp_path: Path) -> None:
    tsv = _write_bindingdb_tsv(
        tmp_path / "BindingDB_All.tsv",
        [
            "CCN\tBSYNRYMUTXBXSQ-UHFFFAOYSA-M\tMouse\tP11111\tTYR\t\t\t1\t10.1000/test",
        ],
    )
    out_csv = tmp_path / "out.csv"

    res = run_script(
        tmp_path,
        [
            "--bindingdb-tsv",
            str(tsv),
            "--out-csv",
            str(out_csv),
        ],
    )

    assert res.returncode == 0, res.stderr
    out = pd.read_csv(out_csv)
    assert out.empty


def test_build_panel_include_non_human_replaces_deprecated_skip_non_human(
    tmp_path: Path,
) -> None:
    tsv = _write_bindingdb_tsv(
        tmp_path / "BindingDB_All.tsv",
        [
            "CCN\tBSYNRYMUTXBXSQ-UHFFFAOYSA-M\tMouse\tP11111\tTYR\t\t\t1\t10.1000/test",
        ],
    )
    out_csv = tmp_path / "out.csv"

    res = run_script(
        tmp_path,
        [
            "--bindingdb-tsv",
            str(tsv),
            "--out-csv",
            str(out_csv),
            "--include-non-human",
        ],
    )

    assert res.returncode == 0, res.stderr
    out = pd.read_csv(out_csv)
    assert out.shape[0] == 1
    assert out.loc[0, "known_targets"] == "P11111"


def test_build_panel_preserves_target_aligned_weights(tmp_path: Path) -> None:
    tsv = _write_bindingdb_tsv(
        tmp_path / "BindingDB_All.tsv",
        [
            "CCO\tBSYNRYMUTXBXSQ-UHFFFAOYSA-N\tHomo sapiens\tP11111\tTYR\t\t\t1\t10.1000/strong",
            "CCO\tBSYNRYMUTXBXSQ-UHFFFAOYSA-N\tHomo sapiens\tP22222\tDCT\t\t\t1000000\t10.1000/weak",
        ],
    )
    out_csv = tmp_path / "out.csv"

    res = run_script(
        tmp_path,
        [
            "--bindingdb-tsv",
            str(tsv),
            "--out-csv",
            str(out_csv),
            "--max-targets-per-compound",
            "10",
        ],
    )

    assert res.returncode == 0, res.stderr
    out = pd.read_csv(out_csv)
    assert out.loc[0, "known_targets"] == "P11111;P22222"
    assert out.loc[0, "known_target_weights"] == "1.000000;0.200000"
    assert out.loc[0, "known_target_evidence_notes"] == "10.1000/strong;10.1000/weak"
    assert out.loc[0, "source_weight"] == 1.0


def test_build_panel_max_affinity_excludes_missing_affinity(tmp_path: Path) -> None:
    tsv = _write_bindingdb_tsv(
        tmp_path / "BindingDB_All.tsv",
        [
            "CCO\tBSYNRYMUTXBXSQ-UHFFFAOYSA-N\tHomo sapiens\tP11111\tTYR\t\t\t\t10.1000/missing",
            "CCN\tBSYNRYMUTXBXSQ-UHFFFAOYSA-M\tHomo sapiens\tP22222\tDCT\t\t\t10\t10.1000/present",
        ],
    )
    out_csv = tmp_path / "out.csv"

    res = run_script(
        tmp_path,
        [
            "--bindingdb-tsv",
            str(tsv),
            "--out-csv",
            str(out_csv),
            "--max-affinity-nm",
            "100",
        ],
    )

    assert res.returncode == 0, res.stderr
    out = pd.read_csv(out_csv)
    assert out.shape[0] == 1
    assert out.loc[0, "known_targets"] == "P22222"


@pytest.mark.parametrize(
    ("option", "message"),
    [
        ("--default-source-weight", "--default-source-weight must be in [0, 1]"),
        ("--min-source-weight", "--min-source-weight must be in [0, 1]"),
        ("--max-affinity-nm", "--max-affinity-nm must be > 0 and finite"),
    ],
)
def test_build_panel_rejects_nonfinite_numeric_options(
    tmp_path: Path,
    option: str,
    message: str,
) -> None:
    tsv = _write_bindingdb_tsv(
        tmp_path / "BindingDB_All.tsv",
        [
            "CCO\tBSYNRYMUTXBXSQ-UHFFFAOYSA-N\tHomo sapiens\tP11111\tTYR\t\t\t1\t10.1000/test",
        ],
    )
    out_csv = tmp_path / "out.csv"
    out_csv.write_text("stale\n")

    res = run_script(
        tmp_path,
        [
            "--bindingdb-tsv",
            str(tsv),
            "--out-csv",
            str(out_csv),
            option,
            "nan",
        ],
    )

    assert res.returncode != 0
    assert message in res.stderr
    assert not out_csv.exists()


def test_build_panel_fails_closed_without_required_target_chain_id_header(
    tmp_path: Path,
) -> None:
    tsv = tmp_path / "BindingDB_All.tsv"
    tsv.write_text(
        "Ligand SMILES\tLigand InChI Key\tTarget Source Organism According to Curator or DataSource\t"
        "Ki (nM)\tBindingDB Entry DOI\n"
        "CCO\tBSYNRYMUTXBXSQ-UHFFFAOYSA-N\tHomo sapiens\t1\t10.1000/test\n",
        encoding="utf-8",
    )
    out_csv = tmp_path / "out.csv"
    out_csv.write_text("stale\n")

    res = run_script(
        tmp_path,
        [
            "--bindingdb-tsv",
            str(tsv),
            "--out-csv",
            str(out_csv),
        ],
    )

    assert res.returncode != 0
    assert "missing required target chain UniProt ID column" in res.stderr
    assert not out_csv.exists()


def test_build_panel_writes_summary_atomically_and_creates_parent(
    tmp_path: Path,
) -> None:
    tsv = _write_bindingdb_tsv(
        tmp_path / "BindingDB_All.tsv",
        [
            "CCO\tBSYNRYMUTXBXSQ-UHFFFAOYSA-N\tHomo sapiens\tP11111\tTYR\t\t\t1\t10.1000/test",
        ],
    )
    summary = tmp_path / "nested" / "summary.json"

    res = run_script(
        tmp_path,
        [
            "--bindingdb-tsv",
            str(tsv),
            "--out-csv",
            str(tmp_path / "nested" / "out.csv"),
            "--out-summary-json",
            str(summary),
        ],
    )

    assert res.returncode == 0, res.stderr
    assert summary.exists()
