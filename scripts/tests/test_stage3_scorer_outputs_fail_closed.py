"""Regression tests for Stage 3 scorer output cleanup."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]


def run_script(
    tmp_path: Path,
    script: str,
    args: list[str],
    out_name: str,
    *,
    empty_path: bool = False,
) -> subprocess.CompletedProcess[str]:
    out = tmp_path / out_name
    out.write_text("stale\n")
    env = os.environ.copy()
    if empty_path:
        empty = tmp_path / f"{script}_empty_path"
        empty.mkdir(exist_ok=True)
        env["PATH"] = str(empty)
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def write_top_csv(path: Path) -> None:
    pd.DataFrame([{"target_id": "P1"}]).to_csv(path, index=False)


def write_clean_receptor(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\n"
        "END\n"
    )


def load_fast_rerank_module(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "stage3_fast_rerank_under_test",
        ROOT / "scripts/stage3_fast_rerank.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fast_rerank_budget_never_underfills_requested_top_n(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_fast_rerank_module(monkeypatch)

    requested, effective = module._effective_candidate_count(
        autodock_count=10_000,
        rerank_candidates=10,
        top_n=50,
    )

    assert requested == 10
    assert effective == 50


def test_fast_rerank_auto_budget_keeps_larger_one_percent_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_fast_rerank_module(monkeypatch)

    requested, effective = module._effective_candidate_count(
        autodock_count=20_000,
        rerank_candidates=0,
        top_n=50,
    )

    assert requested == 200
    assert effective == 200


def test_fast_rerank_auto_budget_rounds_one_percent_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_fast_rerank_module(monkeypatch)

    requested, effective = module._effective_candidate_count(
        autodock_count=5_050,
        rerank_candidates=0,
        top_n=50,
    )

    assert requested == 51
    assert effective == 51


def test_fast_rerank_cli_auto_budget_preserves_one_percent_before_top_n(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_fast_rerank_module(monkeypatch)
    autodock = tmp_path / "autodock.tsv"
    ligand = tmp_path / "ligand.sdf"
    clean = tmp_path / "clean"
    out = tmp_path / "fast.csv"
    clean.mkdir()
    pd.DataFrame(
        {
            "target_id": [f"P{index:05d}" for index in range(6_000)],
            "neg_vina_score": [float(6_000 - index) for index in range(6_000)],
        }
    ).to_csv(autodock, sep="\t", index=False)
    ligand.write_text("ligand\n")
    reranked: list[str] = []
    for index in range(60):
        target_id = f"P{index:05d}"
        write_clean_receptor(clean / f"{target_id}_clean.pdb")

    def fake_gnina(receptor: Path, _ligand: Path) -> float:
        reranked.append(receptor.stem.removesuffix("_clean"))
        return float(6_000 - int(reranked[-1][1:]))

    monkeypatch.setattr(module, "gnina_score", fake_gnina)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(ROOT / "scripts/stage3_fast_rerank.py"),
            "--autodock-scores",
            str(autodock),
            "--ligand-sdf",
            str(ligand),
            "--clean-dir",
            str(clean),
            "--rerank-candidates",
            "0",
            "--rtm-top-n",
            "0",
            "--boltz-top-n",
            "0",
            "--top-n",
            "50",
            "--out-csv",
            str(out),
        ],
    )

    module.main()

    result = pd.read_csv(out)
    assert reranked == [f"P{index:05d}" for index in range(60)]
    assert len(result) == 50
    assert result["target_id"].tolist() == [f"P{index:05d}" for index in range(50)]


def run_fast_rerank(
    tmp_path: Path,
    autodock: Path,
    ligand: Path,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    out = tmp_path / "fast.csv"
    out.write_text("stale\n")
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_fast_rerank.py"),
            "--autodock-scores", str(autodock),
            "--ligand-sdf", str(ligand),
            "--clean-dir", str(tmp_path / "clean"),
            "--out-csv", str(out),
            *(extra_args or []),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def run_fast_rerank_in_process(
    module,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    autodock = tmp_path / "autodock.tsv"
    ligand = tmp_path / "ligand.sdf"
    out = tmp_path / "fast.csv"
    clean = tmp_path / "clean"
    autodock.write_text("target_id\tneg_vina_score\nP1\t3.0\n")
    ligand.write_text("ligand\n")
    write_clean_receptor(clean / "P1_clean.pdb")
    out.write_text("stale\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(ROOT / "scripts/stage3_fast_rerank.py"),
            "--autodock-scores", str(autodock),
            "--ligand-sdf", str(ligand),
            "--clean-dir", str(clean),
            "--rerank-candidates", "1",
            "--out-csv", str(out),
        ],
    )
    return out


def test_fast_rerank_empty_autodock_removes_stale_output(tmp_path: Path) -> None:
    autodock = tmp_path / "autodock.tsv"
    ligand = tmp_path / "ligand.sdf"
    autodock.write_text("")
    ligand.write_text("ligand\n")

    res = run_fast_rerank(tmp_path, autodock, ligand)

    assert res.returncode != 0
    assert "AutoDock score file is missing or empty for fast rerank" in res.stderr
    assert not (tmp_path / "fast.csv").exists()


def test_fast_rerank_rejects_missing_autodock_columns(tmp_path: Path) -> None:
    autodock = tmp_path / "autodock.tsv"
    ligand = tmp_path / "ligand.sdf"
    autodock.write_text("target_id\tother\nP1\t1.0\n")
    ligand.write_text("ligand\n")

    res = run_fast_rerank(tmp_path, autodock, ligand)

    assert res.returncode != 0
    assert "AutoDock score file missing required columns for fast rerank" in res.stderr
    assert not (tmp_path / "fast.csv").exists()


def test_fast_rerank_rejects_non_numeric_autodock_scores(tmp_path: Path) -> None:
    autodock = tmp_path / "autodock.tsv"
    ligand = tmp_path / "ligand.sdf"
    autodock.write_text("target_id\tneg_vina_score\nP1\tnot-a-number\n")
    ligand.write_text("ligand\n")

    res = run_fast_rerank(tmp_path, autodock, ligand)

    assert res.returncode != 0
    assert "AutoDock score file column 'neg_vina_score' contains invalid values" in res.stderr
    assert not (tmp_path / "fast.csv").exists()


def test_fast_rerank_rejects_boolean_autodock_scores(tmp_path: Path) -> None:
    autodock = tmp_path / "autodock.tsv"
    ligand = tmp_path / "ligand.sdf"
    autodock.write_text("target_id\tneg_vina_score\nP1\tTrue\n")
    ligand.write_text("ligand\n")

    res = run_fast_rerank(tmp_path, autodock, ligand)

    assert res.returncode != 0
    assert "AutoDock score file column 'neg_vina_score' contains invalid values" in res.stderr
    assert not (tmp_path / "fast.csv").exists()


def test_fast_rerank_rejects_partially_invalid_autodock_scores(
    tmp_path: Path,
) -> None:
    autodock = tmp_path / "autodock.tsv"
    ligand = tmp_path / "ligand.sdf"
    autodock.write_text("target_id\tneg_vina_score\nP1\t3.0\nP2\tnot-a-number\n")
    ligand.write_text("ligand\n")

    res = run_fast_rerank(tmp_path, autodock, ligand)

    assert res.returncode != 0
    assert "AutoDock score file column 'neg_vina_score' contains invalid values" in res.stderr
    assert not (tmp_path / "fast.csv").exists()


def test_fast_rerank_rejects_blank_autodock_target_ids(tmp_path: Path) -> None:
    autodock = tmp_path / "autodock.tsv"
    ligand = tmp_path / "ligand.sdf"
    autodock.write_text("target_id\tneg_vina_score\nP1\t3.0\n \t2.0\n")
    ligand.write_text("ligand\n")

    res = run_fast_rerank(tmp_path, autodock, ligand)

    assert res.returncode != 0
    assert "AutoDock score file column 'target_id' contains blank values" in res.stderr
    assert not (tmp_path / "fast.csv").exists()


def test_fast_rerank_rejects_duplicate_autodock_target_ids(
    tmp_path: Path,
) -> None:
    autodock = tmp_path / "autodock.tsv"
    ligand = tmp_path / "ligand.sdf"
    autodock.write_text("target_id\tneg_vina_score\nP1\t3.0\nP1\t2.0\n")
    ligand.write_text("ligand\n")

    res = run_fast_rerank(tmp_path, autodock, ligand)

    assert res.returncode != 0
    assert (
        "AutoDock score file contains duplicate target_id values for fast "
        "rerank: P1"
    ) in res.stderr
    assert not (tmp_path / "fast.csv").exists()


def test_fast_rerank_rejects_empty_ligand(tmp_path: Path) -> None:
    autodock = tmp_path / "autodock.tsv"
    ligand = tmp_path / "ligand.sdf"
    autodock.write_text("target_id\tneg_vina_score\nP1\t3.0\n")
    ligand.write_text("")

    res = run_fast_rerank(tmp_path, autodock, ligand)

    assert res.returncode != 0
    assert "Ligand SDF is missing or empty for fast rerank" in res.stderr
    assert not (tmp_path / "fast.csv").exists()


def test_fast_rerank_rejects_no_efficacy_kg_overlap(tmp_path: Path) -> None:
    autodock = tmp_path / "autodock.tsv"
    ligand = tmp_path / "ligand.sdf"
    kg = tmp_path / "skin_efficacy.graphml"
    autodock.write_text("target_id\tneg_vina_score\nP1\t3.0\n")
    ligand.write_text("ligand\n")
    kg.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n'
        '  <graph edgedefault="directed">\n'
        '    <node id="gene:P2" />\n'
        "  </graph>\n"
        "</graphml>\n"
    )

    res = run_fast_rerank(
        tmp_path,
        autodock,
        ligand,
        ["--require-efficacy-kg", str(kg)],
    )

    assert res.returncode != 0
    assert "No AutoDock candidates overlap skin-efficacy KG gene:* targets" in res.stderr
    assert not (tmp_path / "fast.csv").exists()


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--rrf-k", "0"], "--rrf-k must be >= 1: 0"),
        (
            ["--min-sources-per-target", "0"],
            "--min-sources-per-target must be >= 1: 0",
        ),
        (["--top-n", "0"], "--top-n must be >= 1: 0"),
        (
            ["--rerank-candidates", "-1"],
            "--rerank-candidates must be >= 0: -1",
        ),
        (["--rtm-top-n", "-1"], "--rtm-top-n must be >= 0: -1"),
        (["--boltz-top-n", "-1"], "--boltz-top-n must be >= 0: -1"),
        (["--max-residues", "0"], "--max-residues must be >= 1: 0"),
        (
            ["--crop-radius", "0"],
            "--crop-radius must be a finite value > 0: 0",
        ),
        (
            ["--crop-radius", "nan"],
            "--crop-radius must be a finite value > 0: nan",
        ),
    ],
)
def test_fast_rerank_rejects_invalid_numeric_args_without_output(
    tmp_path: Path,
    args: list[str],
    message: str,
) -> None:
    autodock = tmp_path / "autodock.tsv"
    ligand = tmp_path / "ligand.sdf"
    autodock.write_text("target_id\tneg_vina_score\nP1\t3.0\n")
    ligand.write_text("ligand\n")

    res = run_fast_rerank(tmp_path, autodock, ligand, args)

    assert res.returncode != 0
    assert message in res.stderr
    assert not (tmp_path / "fast.csv").exists()


def test_fast_rerank_rejects_nonfinite_gnina_score_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_fast_rerank_module(monkeypatch)
    out = run_fast_rerank_in_process(module, tmp_path, monkeypatch)
    monkeypatch.setattr(module, "gnina_score", lambda *_args: float("nan"))
    monkeypatch.setattr(module, "rtmscore", lambda *_args: 2.0)
    monkeypatch.setattr(module, "call_boltz", lambda *_args: 2.0)

    with pytest.raises(SystemExit) as excinfo:
        module.main()

    assert "GNINA returned non-finite score for P1" in str(excinfo.value)
    assert not out.exists()


def test_fast_rerank_rejects_nonfinite_rtmscore_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_fast_rerank_module(monkeypatch)
    out = run_fast_rerank_in_process(module, tmp_path, monkeypatch)
    monkeypatch.setattr(module, "gnina_score", lambda *_args: 2.0)
    monkeypatch.setattr(module, "rtmscore", lambda *_args: float("inf"))
    monkeypatch.setattr(module, "call_boltz", lambda *_args: 2.0)

    with pytest.raises(SystemExit) as excinfo:
        module.main()

    assert "RTMScore returned non-finite score for P1" in str(excinfo.value)
    assert not out.exists()


def test_fast_rerank_rejects_nonfinite_boltz_score_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_fast_rerank_module(monkeypatch)
    out = run_fast_rerank_in_process(module, tmp_path, monkeypatch)
    monkeypatch.setattr(module, "gnina_score", lambda *_args: 2.0)
    monkeypatch.setattr(module, "rtmscore", lambda *_args: 2.0)
    monkeypatch.setattr(module, "call_boltz", lambda *_args: float("-inf"))

    with pytest.raises(SystemExit) as excinfo:
        module.main()

    assert "Boltz-2 returned non-finite score for P1" in str(excinfo.value)
    assert not out.exists()


def test_gnina_missing_binary_removes_stale_scores(tmp_path: Path) -> None:
    top = tmp_path / "top.csv"
    write_top_csv(top)

    res = run_script(
        tmp_path,
        "stage3_gnina_rescore.py",
        [
            "--top-csv", str(top),
            "--ligand-sdf", str(tmp_path / "ligand.sdf"),
            "--clean-dir", str(tmp_path / "clean"),
            "--out-scores", str(tmp_path / "gnina.tsv"),
        ],
        "gnina.tsv",
        empty_path=True,
    )

    assert res.returncode != 0
    assert "gnina is not available" in res.stderr
    assert not (tmp_path / "gnina.tsv").exists()


def test_gnina_empty_ligand_removes_stale_scores(tmp_path: Path) -> None:
    top = tmp_path / "top.csv"
    write_top_csv(top)
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "P1_clean.pdb").write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\nEND\n"
    )
    ligand = tmp_path / "ligand.sdf"
    ligand.write_text("")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gnina = fake_bin / "gnina"
    fake_gnina.write_text("#!/bin/sh\nprintf 'CNNaffinity: 1.5\\n'\n")
    fake_gnina.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', os.defpath)}"
    (tmp_path / "gnina.tsv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_gnina_rescore.py"),
            "--top-csv", str(top),
            "--ligand-sdf", str(ligand),
            "--clean-dir", str(clean),
            "--out-scores", str(tmp_path / "gnina.tsv"),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "Ligand SDF is missing or empty for GNINA" in res.stderr
    assert not (tmp_path / "gnina.tsv").exists()


def test_gnina_uses_cpu_rescore_mode_and_writes_scores(tmp_path: Path) -> None:
    top = tmp_path / "top.csv"
    write_top_csv(top)
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "P1_clean.pdb").write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\nEND\n"
    )
    ligand = tmp_path / "ligand.sdf"
    ligand.write_text("ligand\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    args_file = tmp_path / "gnina_args.txt"
    fake_gnina = fake_bin / "gnina"
    fake_gnina.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$@\" > {args_file}\n"
        "printf 'CNNaffinity: 1.5\\n'\n"
    )
    fake_gnina.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', os.defpath)}"
    (tmp_path / "gnina.tsv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_gnina_rescore.py"),
            "--top-csv", str(top),
            "--ligand-sdf", str(ligand),
            "--clean-dir", str(clean),
            "--out-scores", str(tmp_path / "gnina.tsv"),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode == 0, res.stderr
    assert "--no_gpu" in args_file.read_text().splitlines()
    assert "rescore" in args_file.read_text().splitlines()
    assert (tmp_path / "gnina.tsv").read_text() == (
        "target_id\tcnn_affinity\tscore\nP1\t1.5000\t1.5000\n"
    )


def test_gnina_nonfinite_score_removes_stale_scores(tmp_path: Path) -> None:
    top = tmp_path / "top.csv"
    write_top_csv(top)
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "P1_clean.pdb").write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\nEND\n"
    )
    ligand = tmp_path / "ligand.sdf"
    ligand.write_text("ligand\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gnina = fake_bin / "gnina"
    fake_gnina.write_text("#!/bin/sh\nprintf 'CNNaffinity: nan\\n'\n")
    fake_gnina.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', os.defpath)}"
    (tmp_path / "gnina.tsv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_gnina_rescore.py"),
            "--top-csv", str(top),
            "--ligand-sdf", str(ligand),
            "--clean-dir", str(clean),
            "--out-scores", str(tmp_path / "gnina.tsv"),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "GNINA output for P1_clean.pdb must be finite" in res.stderr
    assert not (tmp_path / "gnina.tsv").exists()
    assert not (tmp_path / "gnina.tsv.tmp").exists()


def test_gnina_rejects_partial_nonfinite_scores_without_output(
    tmp_path: Path,
) -> None:
    top = tmp_path / "top.csv"
    pd.DataFrame([{"target_id": "P1"}, {"target_id": "P2"}]).to_csv(
        top,
        index=False,
    )
    clean = tmp_path / "clean"
    clean.mkdir()
    for target_id in ("P1", "P2"):
        (clean / f"{target_id}_clean.pdb").write_text(
            "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\nEND\n"
        )
    ligand = tmp_path / "ligand.sdf"
    ligand.write_text("ligand\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gnina = fake_bin / "gnina"
    fake_gnina.write_text(
        "#!/bin/sh\n"
        "case \"$*\" in\n"
        "  *P2_clean.pdb*) printf 'CNNaffinity: inf\\n' ;;\n"
        "  *) printf 'CNNaffinity: 1.5\\n' ;;\n"
        "esac\n"
    )
    fake_gnina.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', os.defpath)}"
    (tmp_path / "gnina.tsv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_gnina_rescore.py"),
            "--top-csv", str(top),
            "--ligand-sdf", str(ligand),
            "--clean-dir", str(clean),
            "--out-scores", str(tmp_path / "gnina.tsv"),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "GNINA output for P2_clean.pdb must be finite" in res.stderr
    assert not (tmp_path / "gnina.tsv").exists()
    assert not (tmp_path / "gnina.tsv.tmp").exists()


@pytest.mark.parametrize(
    ("script", "out_name", "label"),
    [
        ("stage3_gnina_rescore.py", "gnina.tsv", "GNINA"),
        ("stage3_rtmscore.py", "rtm.tsv", "RTMScore"),
    ],
)
@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (None, "top-target CSV is required and must be non-empty"),
        ([{"score": 1.0}], "top-target CSV missing required column 'target_id'"),
        (
            [{"target_id": "P1"}, {"target_id": ""}],
            "top-target CSV column 'target_id' contains blank values",
        ),
        (
            [{"target_id": "P1"}, {"target_id": "P1"}],
            "top-target CSV contains duplicate target_id values: P1",
        ),
    ],
)
def test_stage3_rescorers_reject_invalid_top_csv_without_output(
    tmp_path: Path,
    script: str,
    out_name: str,
    label: str,
    rows: list[dict[str, object]] | None,
    message: str,
) -> None:
    top = tmp_path / "top.csv"
    if rows is None:
        top.write_text("")
    else:
        pd.DataFrame(rows).to_csv(top, index=False)

    res = run_script(
        tmp_path,
        script,
        [
            "--top-csv", str(top),
            "--ligand-sdf", str(tmp_path / "ligand.sdf"),
            "--clean-dir", str(tmp_path / "clean"),
            "--out-scores", str(tmp_path / out_name),
        ],
        out_name,
    )

    assert res.returncode != 0
    assert f"{label} {message}" in res.stderr
    assert not (tmp_path / out_name).exists()


def test_rtmscore_missing_package_removes_stale_scores(tmp_path: Path) -> None:
    top = tmp_path / "top.csv"
    write_top_csv(top)
    fake_pkg = tmp_path / "fake_pkg"
    rtmscore_pkg = fake_pkg / "rtmscore"
    rtmscore_pkg.mkdir(parents=True)
    (rtmscore_pkg / "__init__.py").write_text("raise ImportError('forced missing')\n")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(fake_pkg)
    (tmp_path / "rtm.tsv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_rtmscore.py"),
            "--top-csv", str(top),
            "--ligand-sdf", str(tmp_path / "ligand.sdf"),
            "--clean-dir", str(tmp_path / "clean"),
            "--out-scores", str(tmp_path / "rtm.tsv"),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "rtmscore is not installed" in res.stderr
    assert not (tmp_path / "rtm.tsv").exists()


def test_rtmscore_empty_ligand_removes_stale_scores(tmp_path: Path) -> None:
    top = tmp_path / "top.csv"
    write_top_csv(top)
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "P1_clean.pdb").write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\nEND\n"
    )
    ligand = tmp_path / "ligand.sdf"
    ligand.write_text("")
    fake_pkg = tmp_path / "fake_pkg"
    rtmscore_pkg = fake_pkg / "rtmscore"
    rtmscore_pkg.mkdir(parents=True)
    (rtmscore_pkg / "__init__.py").write_text(
        "def predict_affinity(receptor, ligand):\n    return 2.5\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(fake_pkg)
    (tmp_path / "rtm.tsv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_rtmscore.py"),
            "--top-csv", str(top),
            "--ligand-sdf", str(ligand),
            "--clean-dir", str(clean),
            "--out-scores", str(tmp_path / "rtm.tsv"),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "Ligand SDF is missing or empty for RTMScore" in res.stderr
    assert not (tmp_path / "rtm.tsv").exists()


def test_rtmscore_nonfinite_score_removes_stale_scores(tmp_path: Path) -> None:
    top = tmp_path / "top.csv"
    write_top_csv(top)
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "P1_clean.pdb").write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\nEND\n"
    )
    ligand = tmp_path / "ligand.sdf"
    ligand.write_text("ligand\n")
    fake_pkg = tmp_path / "fake_pkg"
    rtmscore_pkg = fake_pkg / "rtmscore"
    rtmscore_pkg.mkdir(parents=True)
    (rtmscore_pkg / "__init__.py").write_text(
        "def predict_affinity(receptor, ligand):\n    return float('inf')\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(fake_pkg)
    (tmp_path / "rtm.tsv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_rtmscore.py"),
            "--top-csv", str(top),
            "--ligand-sdf", str(ligand),
            "--clean-dir", str(clean),
            "--out-scores", str(tmp_path / "rtm.tsv"),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "RTMScore output for P1_clean.pdb must be finite" in res.stderr
    assert not (tmp_path / "rtm.tsv").exists()
    assert not (tmp_path / "rtm.tsv.tmp").exists()


def test_rtmscore_rejects_partial_nonfinite_scores_without_output(
    tmp_path: Path,
) -> None:
    top = tmp_path / "top.csv"
    pd.DataFrame([{"target_id": "P1"}, {"target_id": "P2"}]).to_csv(
        top,
        index=False,
    )
    clean = tmp_path / "clean"
    clean.mkdir()
    for target_id in ("P1", "P2"):
        (clean / f"{target_id}_clean.pdb").write_text(
            "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\nEND\n"
        )
    ligand = tmp_path / "ligand.sdf"
    ligand.write_text("ligand\n")
    fake_pkg = tmp_path / "fake_pkg"
    rtmscore_pkg = fake_pkg / "rtmscore"
    rtmscore_pkg.mkdir(parents=True)
    (rtmscore_pkg / "__init__.py").write_text(
        "def predict_affinity(receptor, ligand):\n"
        "    return float('inf') if 'P2_clean.pdb' in receptor else 2.5\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(fake_pkg)
    (tmp_path / "rtm.tsv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_rtmscore.py"),
            "--top-csv", str(top),
            "--ligand-sdf", str(ligand),
            "--clean-dir", str(clean),
            "--out-scores", str(tmp_path / "rtm.tsv"),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "RTMScore output for P2_clean.pdb must be finite" in res.stderr
    assert not (tmp_path / "rtm.tsv").exists()
    assert not (tmp_path / "rtm.tsv.tmp").exists()


def test_boltz_missing_binary_removes_stale_scores(tmp_path: Path) -> None:
    top = tmp_path / "top.csv"
    write_top_csv(top)

    res = run_script(
        tmp_path,
        "stage3_boltz2_affinity.py",
        [
            "--top-csv", str(top),
            "--ligand-sdf", str(tmp_path / "ligand.sdf"),
            "--clean-dir", str(tmp_path / "clean"),
            "--out-scores", str(tmp_path / "boltz.tsv"),
        ],
        "boltz.tsv",
        empty_path=True,
    )

    assert res.returncode != 0
    assert "boltz is not available" in res.stderr
    assert not (tmp_path / "boltz.tsv").exists()


def test_boltz_empty_ligand_removes_stale_scores(tmp_path: Path) -> None:
    top = tmp_path / "top.csv"
    write_top_csv(top)
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "P1_clean.pdb").write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\nEND\n"
    )
    ligand = tmp_path / "ligand.sdf"
    ligand.write_text("")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_boltz = fake_bin / "boltz"
    fake_boltz.write_text(
        "#!/bin/sh\n"
        "out=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = '--out_dir' ]; then shift; out=\"$1\"; fi\n"
        "  shift\n"
        "done\n"
        "mkdir -p \"$out\"\n"
        "printf '{\"log_uM_affinity\": 1.5}\\n' > \"$out/affinity.json\"\n"
    )
    fake_boltz.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', os.defpath)}"
    (tmp_path / "boltz.tsv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_boltz2_affinity.py"),
            "--top-csv", str(top),
            "--ligand-sdf", str(ligand),
            "--clean-dir", str(clean),
            "--out-scores", str(tmp_path / "boltz.tsv"),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "Ligand SDF is missing or empty for Boltz-2 affinity" in res.stderr
    assert not (tmp_path / "boltz.tsv").exists()


def test_boltz_nonfinite_affinity_json_removes_stale_scores(tmp_path: Path) -> None:
    top = tmp_path / "top.csv"
    write_top_csv(top)
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "P1_clean.pdb").write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\nEND\n"
    )
    ligand = tmp_path / "ligand.sdf"
    ligand.write_text("ligand\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_boltz = fake_bin / "boltz"
    fake_boltz.write_text(
        "#!/bin/sh\n"
        "out=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = '--out_dir' ]; then shift; out=\"$1\"; fi\n"
        "  shift\n"
        "done\n"
        "mkdir -p \"$out\"\n"
        "printf '{\"log_uM_affinity\": \"nan\"}\\n' > \"$out/affinity.json\"\n"
    )
    fake_boltz.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', os.defpath)}"
    (tmp_path / "boltz.tsv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_boltz2_affinity.py"),
            "--top-csv", str(top),
            "--ligand-sdf", str(ligand),
            "--clean-dir", str(clean),
            "--out-scores", str(tmp_path / "boltz.tsv"),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert (
        "Boltz-2 affinity payload for P1_clean.pdb produced no valid affinity score"
        in res.stderr
    )
    assert not (tmp_path / "boltz.tsv").exists()
    assert not (tmp_path / "boltz.tsv.tmp").exists()


def test_boltz_boolean_affinity_json_removes_stale_scores(tmp_path: Path) -> None:
    top = tmp_path / "top.csv"
    write_top_csv(top)
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "P1_clean.pdb").write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\nEND\n"
    )
    ligand = tmp_path / "ligand.sdf"
    ligand.write_text("ligand\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_boltz = fake_bin / "boltz"
    fake_boltz.write_text(
        "#!/bin/sh\n"
        "out=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = '--out_dir' ]; then shift; out=\"$1\"; fi\n"
        "  shift\n"
        "done\n"
        "mkdir -p \"$out\"\n"
        "printf '{\"log_uM_affinity\": true}\\n' > \"$out/affinity.json\"\n"
    )
    fake_boltz.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', os.defpath)}"
    (tmp_path / "boltz.tsv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_boltz2_affinity.py"),
            "--top-csv", str(top),
            "--ligand-sdf", str(ligand),
            "--clean-dir", str(clean),
            "--out-scores", str(tmp_path / "boltz.tsv"),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert (
        "Boltz-2 affinity payload for P1_clean.pdb produced no valid affinity score"
        in res.stderr
    )
    assert not (tmp_path / "boltz.tsv").exists()
    assert not (tmp_path / "boltz.tsv.tmp").exists()


def test_boltz_empty_affinity_json_removes_stale_scores(tmp_path: Path) -> None:
    top = tmp_path / "top.csv"
    write_top_csv(top)
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "P1_clean.pdb").write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\nEND\n"
    )
    ligand = tmp_path / "ligand.sdf"
    ligand.write_text("ligand\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_boltz = fake_bin / "boltz"
    fake_boltz.write_text(
        "#!/bin/sh\n"
        "out=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = '--out_dir' ]; then shift; out=\"$1\"; fi\n"
        "  shift\n"
        "done\n"
        "mkdir -p \"$out\"\n"
        ": > \"$out/affinity.json\"\n"
    )
    fake_boltz.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', os.defpath)}"
    (tmp_path / "boltz.tsv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_boltz2_affinity.py"),
            "--top-csv", str(top),
            "--ligand-sdf", str(ligand),
            "--clean-dir", str(clean),
            "--out-scores", str(tmp_path / "boltz.tsv"),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "Boltz-2 produced no usable affinity scores" in res.stderr
    assert not (tmp_path / "boltz.tsv").exists()
    assert not (tmp_path / "boltz.tsv.tmp").exists()


def test_boltz_rejects_partial_invalid_affinity_scores_without_output(
    tmp_path: Path,
) -> None:
    top = tmp_path / "top.csv"
    pd.DataFrame([{"target_id": "P1"}, {"target_id": "P2"}]).to_csv(
        top,
        index=False,
    )
    clean = tmp_path / "clean"
    clean.mkdir()
    for target_id in ("P1", "P2"):
        (clean / f"{target_id}_clean.pdb").write_text(
            "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\nEND\n"
        )
    ligand = tmp_path / "ligand.sdf"
    ligand.write_text("CCO\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_boltz = fake_bin / "boltz"
    fake_boltz.write_text(
        "#!/bin/sh\n"
        "out=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = '--out_dir' ]; then shift; out=\"$1\"; fi\n"
        "  shift\n"
        "done\n"
        "mkdir -p \"$out\"\n"
        "case \"$out\" in\n"
        "  *P2*) printf '{\"log_uM_affinity\": \"inf\"}\\n' > \"$out/affinity.json\" ;;\n"
        "  *) printf '{\"log_uM_affinity\": 1.5}\\n' > \"$out/affinity.json\" ;;\n"
        "esac\n"
    )
    fake_boltz.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', os.defpath)}"
    (tmp_path / "boltz.tsv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_boltz2_affinity.py"),
            "--top-csv", str(top),
            "--ligand-sdf", str(ligand),
            "--clean-dir", str(clean),
            "--out-scores", str(tmp_path / "boltz.tsv"),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert (
        "Boltz-2 affinity payload for P2_clean.pdb produced no valid affinity score"
        in res.stderr
    )
    assert not (tmp_path / "boltz.tsv").exists()
    assert not (tmp_path / "boltz.tsv.tmp").exists()


def test_boltz_invalid_affinity_json_removes_stale_scores(tmp_path: Path) -> None:
    top = tmp_path / "top.csv"
    write_top_csv(top)
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "P1_clean.pdb").write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\nEND\n"
    )
    ligand = tmp_path / "ligand.sdf"
    ligand.write_text("ligand\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_boltz = fake_bin / "boltz"
    fake_boltz.write_text(
        "#!/bin/sh\n"
        "out=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = '--out_dir' ]; then shift; out=\"$1\"; fi\n"
        "  shift\n"
        "done\n"
        "mkdir -p \"$out\"\n"
        "printf 'not-json\\n' > \"$out/affinity.json\"\n"
    )
    fake_boltz.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', os.defpath)}"
    (tmp_path / "boltz.tsv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_boltz2_affinity.py"),
            "--top-csv", str(top),
            "--ligand-sdf", str(ligand),
            "--clean-dir", str(clean),
            "--out-scores", str(tmp_path / "boltz.tsv"),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "Boltz-2 produced no usable affinity scores" in res.stderr
    assert not (tmp_path / "boltz.tsv").exists()


def test_boltz_reads_current_affinity_json_shape(tmp_path: Path) -> None:
    top = tmp_path / "top.csv"
    write_top_csv(top)
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "P1_clean.pdb").write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\nEND\n"
    )
    ligand = tmp_path / "ligand.sdf"
    ligand.write_text("CCO\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_boltz = fake_bin / "boltz"
    fake_boltz.write_text(
        "#!/bin/sh\n"
        "out=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = '--out_dir' ]; then shift; out=\"$1\"; fi\n"
        "  shift\n"
        "done\n"
        "mkdir -p \"$out/predictions/P1\"\n"
        "printf '{\"affinity_pred_value\": 1.5, \"affinity_probability_binary\": 0.81}\\n' > \"$out/predictions/P1/affinity_P1.json\"\n"
    )
    fake_boltz.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', os.defpath)}"

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_boltz2_affinity.py"),
            "--top-csv", str(top),
            "--ligand-sdf", str(ligand),
            "--clean-dir", str(clean),
            "--out-scores", str(tmp_path / "boltz.tsv"),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode == 0, res.stderr
    rows = pd.read_csv(tmp_path / "boltz.tsv", sep="\t")
    assert rows["target_id"].tolist() == ["P1"]
    assert rows["score"].tolist() == [0.81]


def test_boltz_empty_top_csv_removes_stale_scores(tmp_path: Path) -> None:
    top = tmp_path / "top.csv"
    top.write_text("")

    res = run_script(
        tmp_path,
        "stage3_boltz2_affinity.py",
        [
            "--top-csv", str(top),
            "--ligand-sdf", str(tmp_path / "ligand.sdf"),
            "--clean-dir", str(tmp_path / "clean"),
            "--out-scores", str(tmp_path / "boltz.tsv"),
        ],
        "boltz.tsv",
    )

    assert res.returncode != 0
    assert "Boltz-2 top-target CSV is required and must be non-empty" in res.stderr
    assert not (tmp_path / "boltz.tsv").exists()


def test_boltz_top_csv_missing_target_id_removes_stale_scores(tmp_path: Path) -> None:
    top = tmp_path / "top.csv"
    pd.DataFrame([{"score": 1.0}]).to_csv(top, index=False)

    res = run_script(
        tmp_path,
        "stage3_boltz2_affinity.py",
        [
            "--top-csv", str(top),
            "--ligand-sdf", str(tmp_path / "ligand.sdf"),
            "--clean-dir", str(tmp_path / "clean"),
            "--out-scores", str(tmp_path / "boltz.tsv"),
        ],
        "boltz.tsv",
    )

    assert res.returncode != 0
    assert "Boltz-2 top-target CSV missing required column 'target_id'" in res.stderr
    assert not (tmp_path / "boltz.tsv").exists()


def test_boltz_top_csv_duplicate_target_id_removes_stale_scores(tmp_path: Path) -> None:
    top = tmp_path / "top.csv"
    pd.DataFrame([{"target_id": "P1"}, {"target_id": "P1"}]).to_csv(top, index=False)

    res = run_script(
        tmp_path,
        "stage3_boltz2_affinity.py",
        [
            "--top-csv", str(top),
            "--ligand-sdf", str(tmp_path / "ligand.sdf"),
            "--clean-dir", str(tmp_path / "clean"),
            "--out-scores", str(tmp_path / "boltz.tsv"),
        ],
        "boltz.tsv",
    )

    assert res.returncode != 0
    assert (
        "Boltz-2 top-target CSV contains duplicate target_id values: P1"
        in res.stderr
    )
    assert not (tmp_path / "boltz.tsv").exists()


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--max-residues", "0"], "--max-residues must be >= 1: 0"),
        (
            ["--crop-radius", "0"],
            "--crop-radius must be a finite value > 0: 0",
        ),
        (
            ["--crop-radius", "nan"],
            "--crop-radius must be a finite value > 0: nan",
        ),
    ],
)
def test_boltz_rejects_invalid_numeric_args_without_output(
    tmp_path: Path,
    args: list[str],
    message: str,
) -> None:
    top = tmp_path / "top.csv"
    write_top_csv(top)

    res = run_script(
        tmp_path,
        "stage3_boltz2_affinity.py",
        [
            "--top-csv", str(top),
            "--ligand-sdf", str(tmp_path / "ligand.sdf"),
            "--clean-dir", str(tmp_path / "clean"),
            "--out-scores", str(tmp_path / "boltz.tsv"),
            *args,
        ],
        "boltz.tsv",
    )

    assert res.returncode != 0
    assert message in res.stderr
    assert not (tmp_path / "boltz.tsv").exists()


def test_autodock_no_receptors_removes_stale_scores(tmp_path: Path) -> None:
    receptor_dir = tmp_path / "receptors"
    box_dir = tmp_path / "boxes"
    receptor_dir.mkdir()
    box_dir.mkdir()

    res = run_script(
        tmp_path,
        "stage3_autodock_run.py",
        [
            "--ligand-pdbqt", str(tmp_path / "ligand.pdbqt"),
            "--receptor-dir", str(receptor_dir),
            "--box-dir", str(box_dir),
            "--out-scores", str(tmp_path / "autodock.tsv"),
        ],
        "autodock.tsv",
    )

    assert res.returncode != 0
    assert "No receptors selected" in res.stderr
    assert not (tmp_path / "autodock.tsv").exists()


def test_autodock_empty_ligand_removes_stale_scores(tmp_path: Path) -> None:
    receptor_dir = tmp_path / "receptors"
    box_dir = tmp_path / "boxes"
    receptor_dir.mkdir()
    box_dir.mkdir()
    (receptor_dir / "P1.pdbqt").write_text("receptor\n")
    (receptor_dir / "P1.maps.fld").write_text("maps\n")
    (box_dir / "P1.box.txt").write_text(
        "center_x=0\ncenter_y=0\ncenter_z=0\nsize_x=10\nsize_y=10\nsize_z=10\n"
    )
    ligand = tmp_path / "ligand.pdbqt"
    ligand.write_text("")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_autodock = fake_bin / "autodock_gpu_128wi"
    fake_autodock.write_text("#!/bin/sh\nexit 0\n")
    fake_autodock.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = str(fake_bin)
    (tmp_path / "autodock.tsv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_autodock_run.py"),
            "--ligand-pdbqt", str(ligand),
            "--receptor-dir", str(receptor_dir),
            "--box-dir", str(box_dir),
            "--out-scores", str(tmp_path / "autodock.tsv"),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "Ligand PDBQT is missing or empty for AutoDock-GPU" in res.stderr
    assert not (tmp_path / "autodock.tsv").exists()


def test_autodock_empty_dlg_removes_stale_scores(tmp_path: Path) -> None:
    receptor_dir = tmp_path / "receptors"
    box_dir = tmp_path / "boxes"
    receptor_dir.mkdir()
    box_dir.mkdir()
    (receptor_dir / "P1.pdbqt").write_text("receptor\n")
    (receptor_dir / "P1.maps.fld").write_text("maps\n")
    (box_dir / "P1.box.txt").write_text(
        "center_x=0\ncenter_y=0\ncenter_z=0\nsize_x=10\nsize_y=10\nsize_z=10\n"
    )
    ligand = tmp_path / "ligand.pdbqt"
    ligand.write_text("ligand\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_autodock = fake_bin / "autodock_gpu_128wi"
    fake_autodock.write_text(
        "#!/bin/sh\n"
        "res=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = '--resnam' ]; then shift; res=\"$1\"; fi\n"
        "  shift\n"
        "done\n"
        ": > \"${res}.dlg\"\n"
    )
    fake_autodock.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = str(fake_bin)
    (tmp_path / "autodock.tsv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_autodock_run.py"),
            "--ligand-pdbqt", str(ligand),
            "--receptor-dir", str(receptor_dir),
            "--box-dir", str(box_dir),
            "--out-scores", str(tmp_path / "autodock.tsv"),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "AutoDock-GPU produced no usable receptor scores" in res.stderr
    assert not (tmp_path / "autodock.tsv").exists()


def write_pose_export_shims(shim_dir: Path) -> None:
    rdkit_dir = shim_dir / "rdkit"
    rdkit_dir.mkdir(parents=True)
    (rdkit_dir / "__init__.py").write_text("")
    (rdkit_dir / "Chem.py").write_text(
        "class FakeMol:\n"
        "    def __init__(self, pose_text=''):\n"
        "        self.pose_text = pose_text\n"
        "        self.props = {}\n"
        "    def SetProp(self, key, value):\n"
        "        self.props[key] = value\n"
        "\n"
        "class SDWriter:\n"
        "    def __init__(self, path):\n"
        "        self.path = path\n"
        "        self.fh = open(path, 'w')\n"
        "    def write(self, mol):\n"
        "        self.fh.write(mol.props.get('_Name', 'pose') + '\\n')\n"
        "        self.fh.write('  SkinScout\\n\\n')\n"
        "        atoms = [line for line in mol.pose_text.splitlines() if line.startswith(('ATOM', 'HETATM'))]\n"
        "        self.fh.write(f'{len(atoms):>3}  0  0  0  0  0            999 V2000\\n')\n"
        "        for line in atoms:\n"
        "            parts = line.split()\n"
        "            x, y, z = parts[6:9]\n"
        "            self.fh.write(f'{float(x):10.4f}{float(y):10.4f}{float(z):10.4f} C   0  0  0  0  0  0  0  0  0  0  0  0\\n')\n"
        "        self.fh.write('M  END\\n')\n"
        "        for key in sorted(mol.props):\n"
        "            self.fh.write(f'>  <{key}>\\n{mol.props[key]}\\n\\n')\n"
        "        self.fh.write('$$$$\\n')\n"
        "    def close(self):\n"
        "        self.fh.close()\n"
        "\n"
        "class SDMolSupplier:\n"
        "    def __init__(self, path, sanitize=True, removeHs=False):\n"
        "        text = open(path).read() if path else ''\n"
        "        self.mols = [FakeMol(text)] if '$$$$' in text and 'M  END' in text else []\n"
        "    def __iter__(self):\n"
        "        return iter(self.mols)\n"
    )
    (shim_dir / "meeko.py").write_text(
        "class PDBQTMolecule:\n"
        "    def __init__(self, text):\n"
        "        self.text = text\n"
        "    @classmethod\n"
        "    def from_file(cls, path, skip_typing=True):\n"
        "        return cls(open(path).read())\n"
        "\n"
        "class FakeMol:\n"
        "    def __init__(self, pose_text):\n"
        "        self.pose_text = pose_text\n"
        "        self.props = {}\n"
        "    def SetProp(self, key, value):\n"
        "        self.props[key] = value\n"
        "\n"
        "class RDKitMolCreate:\n"
        "    @staticmethod\n"
        "    def from_pdbqt_mol(pdbqt_mol):\n"
        "        return [FakeMol(pdbqt_mol.text)]\n"
    )


def write_basic_autodock_inputs(tmp_path: Path, targets: tuple[str, ...]) -> tuple[Path, Path, Path]:
    receptor_dir = tmp_path / "receptors"
    box_dir = tmp_path / "boxes"
    receptor_dir.mkdir()
    box_dir.mkdir()
    for target in targets:
        (receptor_dir / f"{target}.pdbqt").write_text("receptor\n")
        (receptor_dir / f"{target}.maps.fld").write_text("maps\n")
        (box_dir / f"{target}.box.txt").write_text(
            "center_x=0\ncenter_y=0\ncenter_z=0\nsize_x=10\nsize_y=10\nsize_z=10\n"
        )
    ligand = tmp_path / "ligand.pdbqt"
    ligand.write_text("ligand\n")
    return receptor_dir, box_dir, ligand


def run_autodock_with_pose_shims(
    tmp_path: Path,
    targets: tuple[str, ...],
    autodock_script: str,
    *,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    receptor_dir, box_dir, ligand = write_basic_autodock_inputs(tmp_path, targets)
    fake_bin = tmp_path / "bin"
    shim_dir = tmp_path / "shim"
    fake_bin.mkdir()
    shim_dir.mkdir()
    write_pose_export_shims(shim_dir)
    fake_autodock = fake_bin / "autodock_gpu_128wi"
    fake_autodock.write_text(autodock_script)
    fake_autodock.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = str(fake_bin)
    env["PYTHONPATH"] = str(shim_dir)
    (tmp_path / "autodock.tsv").write_text("stale\n")
    pose_dir = tmp_path / "poses"
    pose_dir.mkdir()
    (pose_dir / "stale.sdf").write_text("stale\n")
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_autodock_run.py"),
            "--ligand-pdbqt", str(ligand),
            "--receptor-dir", str(receptor_dir),
            "--box-dir", str(box_dir),
            "--out-scores", str(tmp_path / "autodock.tsv"),
            "--out-pose-dir", str(pose_dir),
            *(extra_args or []),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_autodock_exports_real_dlg_pose_coordinates(tmp_path: Path) -> None:
    res = run_autodock_with_pose_shims(
        tmp_path,
        ("P1",),
        "#!/bin/sh\n"
        "res=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = '--resnam' ]; then shift; res=\"$1\"; fi\n"
        "  shift\n"
        "done\n"
        "{\n"
        "  printf 'DOCKED: MODEL        1\\n'\n"
        "  printf 'DOCKED: USER    Estimated Free Energy of Binding = -8.0 kcal/mol\\n'\n"
        "  printf 'DOCKED: ROOT\\n'\n"
        "  printf 'DOCKED: ATOM      1  C1  LIG A   1       1.234   2.345   3.456  0.00  0.00    +0.000 C\\n'\n"
        "  printf 'DOCKED: ENDROOT\\n'\n"
        "  printf 'DOCKED: TORSDOF 0\\n'\n"
        "  printf 'DOCKED: ENDMDL\\n'\n"
        "} > \"${res}.dlg\"\n",
    )

    assert res.returncode == 0, res.stderr
    pose = tmp_path / "poses" / "P1.sdf"
    assert pose.exists()
    pose_text = pose.read_text()
    assert "    1.2340    2.3450    3.4560 C" in pose_text
    assert ">  <target_id>\nP1" in pose_text
    assert ">  <docking_energy_kcal_mol>\n-8.000" in pose_text
    assert "stale" not in pose_text


def test_autodock_real_meeko_roundtrip_preserves_docked_coordinates(
    tmp_path: Path,
) -> None:
    Chem = pytest.importorskip("rdkit.Chem")
    AllChem = pytest.importorskip("rdkit.Chem.AllChem")
    pytest.importorskip("meeko")
    prepare_ligand = Path(sys.executable).with_name("mk_prepare_ligand.py")
    if not prepare_ligand.is_file():
        pytest.skip("Meeko ligand preparation CLI is unavailable")

    ligand = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert AllChem.EmbedMolecule(ligand, randomSeed=7) == 0
    AllChem.MMFFOptimizeMolecule(ligand)
    ligand_sdf = tmp_path / "ligand.sdf"
    writer = Chem.SDWriter(str(ligand_sdf))
    writer.write(ligand)
    writer.close()
    ligand_pdbqt = tmp_path / "ligand.pdbqt"
    prepared = subprocess.run(
        [str(prepare_ligand), "-i", str(ligand_sdf), "-o", str(ligand_pdbqt)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert prepared.returncode == 0, prepared.stderr
    pdbqt_text = ligand_pdbqt.read_text()
    assert "REMARK SMILES CCO" in pdbqt_text

    dlg = tmp_path / "pose.dlg"
    dlg.write_text(
        "DOCKED: USER    Estimated Free Energy of Binding = -7.25 kcal/mol\n"
        + "".join(f"DOCKED: {line}\n" for line in pdbqt_text.splitlines())
    )
    spec = importlib.util.spec_from_file_location(
        "stage3_autodock_real_meeko_test",
        ROOT / "scripts/stage3_autodock_run.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parsed = module.parse_dlg_best_pose(dlg)
    assert parsed is not None
    energy, pose_text = parsed
    assert energy == -7.25
    assert "REMARK SMILES CCO" in pose_text
    docked_pdbqt = tmp_path / "docked.pdbqt"
    docked_pdbqt.write_text(pose_text)
    pose_sdf = tmp_path / "P_REAL.sdf"
    module._write_sdf_from_pdbqt_pose(
        docked_pdbqt,
        pose_sdf,
        "P_REAL",
        energy,
    )

    pose = next(
        molecule
        for molecule in Chem.SDMolSupplier(
            str(pose_sdf), sanitize=True, removeHs=False
        )
        if molecule is not None
    )
    assert pose.GetProp("target_id") == "P_REAL"
    assert float(pose.GetProp("docking_energy_kcal_mol")) == -7.25
    expected = []
    for line in pdbqt_text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            fields = line.split()
            if not fields[-1].startswith("H"):
                expected.append(
                    (
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    )
                )
    conformer = pose.GetConformer()
    actual = [
        (
            conformer.GetAtomPosition(index).x,
            conformer.GetAtomPosition(index).y,
            conformer.GetAtomPosition(index).z,
        )
        for index in range(pose.GetNumAtoms())
        if pose.GetAtomWithIdx(index).GetAtomicNum() != 1
    ]
    assert actual == pytest.approx(expected, abs=1e-6)


def test_autodock_score_pose_parity_for_successful_targets(tmp_path: Path) -> None:
    res = run_autodock_with_pose_shims(
        tmp_path,
        ("P1", "P2"),
        "#!/bin/sh\n"
        "res=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = '--resnam' ]; then shift; res=\"$1\"; fi\n"
        "  shift\n"
        "done\n"
        "case \"$res\" in */P2) e='-6.5'; x='4.000' ;; *) e='-7.5'; x='1.000' ;; esac\n"
        "{\n"
        "  printf 'DOCKED: MODEL        1\\n'\n"
        "  printf 'DOCKED: USER    Estimated Free Energy of Binding = %s kcal/mol\\n' \"$e\"\n"
        "  printf 'DOCKED: ROOT\\n'\n"
        "  printf 'DOCKED: ATOM      1  C1  LIG A   1       %s   2.000   3.000  0.00  0.00    +0.000 C\\n' \"$x\"\n"
        "  printf 'DOCKED: ENDROOT\\nDOCKED: TORSDOF 0\\nDOCKED: ENDMDL\\n'\n"
        "} > \"${res}.dlg\"\n",
    )

    assert res.returncode == 0, res.stderr
    score_rows = (tmp_path / "autodock.tsv").read_text().splitlines()[1:]
    assert [row.split("\t")[0] for row in score_rows] == ["P1", "P2"]
    assert sorted(path.name for path in (tmp_path / "poses").glob("*.sdf")) == [
        "P1.sdf",
        "P2.sdf",
    ]


def test_autodock_rejects_unsafe_pose_target_id_without_partial_publish(
    tmp_path: Path,
) -> None:
    res = run_autodock_with_pose_shims(
        tmp_path,
        ("P 1",),
        "#!/bin/sh\nexit 1\n",
    )

    assert res.returncode != 0
    assert "Unsafe docking target_id cannot be used as pose filename" in res.stderr
    assert not (tmp_path / "autodock.tsv").exists()
    assert not (tmp_path / "poses").exists()


def test_autodock_pose_failure_removes_scores_and_pose_dir(tmp_path: Path) -> None:
    res = run_autodock_with_pose_shims(
        tmp_path,
        ("P1", "P2"),
        "#!/bin/sh\n"
        "res=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = '--resnam' ]; then shift; res=\"$1\"; fi\n"
        "  shift\n"
        "done\n"
        "case \"$res\" in\n"
        "  */P1) {\n"
        "    printf 'DOCKED: USER    Estimated Free Energy of Binding = -7.0 kcal/mol\\n'\n"
        "    printf 'DOCKED: ROOT\\n'\n"
        "    printf 'DOCKED: ATOM      1  C1  LIG A   1       1.000   2.000   3.000  0.00  0.00    +0.000 C\\n'\n"
        "    printf 'DOCKED: ENDROOT\\nDOCKED: TORSDOF 0\\n'\n"
        "  } > \"${res}.dlg\" ;;\n"
        "  *) printf 'DOCKED: USER    Estimated Free Energy of Binding = -6.0 kcal/mol\\n' > \"${res}.dlg\" ;;\n"
        "esac\n",
    )

    assert res.returncode != 0
    assert "failed to produce scores for all selected receptors" in res.stderr
    assert "1/2 failed [P2]" in res.stderr
    assert not (tmp_path / "autodock.tsv").exists()
    assert not (tmp_path / "autodock.tsv.tmp").exists()
    assert not (tmp_path / "poses").exists()


def test_autodock_default_requires_complete_maps_without_vina_fallback(
    tmp_path: Path,
) -> None:
    receptor_dir = tmp_path / "receptors"
    box_dir = tmp_path / "boxes"
    shim_dir = tmp_path / "shim"
    receptor_dir.mkdir()
    box_dir.mkdir()
    shim_dir.mkdir()
    (receptor_dir / "P1.pdbqt").write_text("receptor\n")
    (box_dir / "P1.box.txt").write_text(
        "center_x=0\ncenter_y=0\ncenter_z=0\nsize_x=10\nsize_y=10\nsize_z=10\n"
    )
    ligand = tmp_path / "ligand.pdbqt"
    ligand.write_text("ligand\n")
    (shim_dir / "vina.py").write_text(
        "raise AssertionError('Vina fallback must not run from the default claim path')\n"
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_autodock = fake_bin / "autodock_gpu_128wi"
    fake_autodock.write_text("#!/bin/sh\nexit 0\n")
    fake_autodock.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = str(fake_bin)
    env["PYTHONPATH"] = str(shim_dir)
    (tmp_path / "autodock.tsv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_autodock_run.py"),
            "--ligand-pdbqt", str(ligand),
            "--receptor-dir", str(receptor_dir),
            "--box-dir", str(box_dir),
            "--out-scores", str(tmp_path / "autodock.tsv"),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "AutoDock-GPU map coverage incomplete: 0/1 receptors" in res.stderr
    assert not (tmp_path / "autodock.tsv").exists()
    assert not (tmp_path / "autodock.tsv.tmp").exists()


def test_autodock_restrict_cap_does_not_replace_missing_ranked_target(
    tmp_path: Path,
) -> None:
    receptor_dir = tmp_path / "receptors"
    box_dir = tmp_path / "boxes"
    receptor_dir.mkdir()
    box_dir.mkdir()
    (receptor_dir / "P1.pdbqt").write_text("receptor\n")
    restrict = tmp_path / "restrict.csv"
    restrict.write_text("target_id\nP2\nP1\n")
    ligand = tmp_path / "ligand.pdbqt"
    ligand.write_text("ligand\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_autodock_run.py"),
            "--ligand-pdbqt", str(ligand),
            "--receptor-dir", str(receptor_dir),
            "--box-dir", str(box_dir),
            "--restrict-list", str(restrict),
            "--max-receptors", "1",
            "--out-scores", str(tmp_path / "autodock.tsv"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "missing non-empty receptor PDBQT files: P2" in res.stderr
    assert not (tmp_path / "autodock.tsv").exists()


def test_autodock_claim_rejects_partial_docking_success(tmp_path: Path) -> None:
    receptor_dir = tmp_path / "receptors"
    box_dir = tmp_path / "boxes"
    receptor_dir.mkdir()
    box_dir.mkdir()
    for target in ("P1", "P2"):
        (receptor_dir / f"{target}.pdbqt").write_text("receptor\n")
        (receptor_dir / f"{target}.maps.fld").write_text("maps\n")
    ligand = tmp_path / "ligand.pdbqt"
    ligand.write_text("ligand\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_autodock = fake_bin / "autodock_gpu_128wi"
    fake_autodock.write_text(
        "#!/bin/sh\n"
        "res=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = '--resnam' ]; then shift; res=\"$1\"; fi\n"
        "  shift\n"
        "done\n"
        "case \"$res\" in\n"
        "  */P1) printf 'DOCKED: USER    Estimated Free Energy of Binding = -7.0 kcal/mol\\n' > \"${res}.dlg\" ;;\n"
        "  *) : > \"${res}.dlg\" ;;\n"
        "esac\n"
    )
    fake_autodock.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = str(fake_bin)

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_autodock_run.py"),
            "--ligand-pdbqt", str(ligand),
            "--receptor-dir", str(receptor_dir),
            "--box-dir", str(box_dir),
            "--out-scores", str(tmp_path / "autodock.tsv"),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "failed to produce scores for all selected receptors" in res.stderr
    assert "1/2 failed [P2]" in res.stderr
    assert not (tmp_path / "autodock.tsv").exists()


def test_autodock_nonfinite_dlg_removes_stale_scores(tmp_path: Path) -> None:
    receptor_dir = tmp_path / "receptors"
    box_dir = tmp_path / "boxes"
    receptor_dir.mkdir()
    box_dir.mkdir()
    (receptor_dir / "P1.pdbqt").write_text("receptor\n")
    (receptor_dir / "P1.maps.fld").write_text("maps\n")
    (box_dir / "P1.box.txt").write_text(
        "center_x=0\ncenter_y=0\ncenter_z=0\nsize_x=10\nsize_y=10\nsize_z=10\n"
    )
    ligand = tmp_path / "ligand.pdbqt"
    ligand.write_text("ligand\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_autodock = fake_bin / "autodock_gpu_128wi"
    fake_autodock.write_text(
        "#!/bin/sh\n"
        "res=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = '--resnam' ]; then shift; res=\"$1\"; fi\n"
        "  shift\n"
        "done\n"
        "printf 'DOCKED: USER    Estimated Free Energy of Binding = inf kcal/mol\\n' > \"${res}.dlg\"\n"
    )
    fake_autodock.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = str(fake_bin)
    (tmp_path / "autodock.tsv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_autodock_run.py"),
            "--ligand-pdbqt", str(ligand),
            "--receptor-dir", str(receptor_dir),
            "--box-dir", str(box_dir),
            "--engine", "autodock_gpu",
            "--out-scores", str(tmp_path / "autodock.tsv"),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "AutoDock-GPU DLG contains non-finite binding energy" in res.stderr
    assert not (tmp_path / "autodock.tsv").exists()
    assert not (tmp_path / "autodock.tsv.tmp").exists()


def test_autodock_vina_engine_writes_scores_from_pdbqt_and_box(
    tmp_path: Path,
) -> None:
    receptor_dir = tmp_path / "receptors"
    box_dir = tmp_path / "boxes"
    shim_dir = tmp_path / "shim"
    receptor_dir.mkdir()
    box_dir.mkdir()
    shim_dir.mkdir()
    (receptor_dir / "P1.pdbqt").write_text("receptor\n")
    (box_dir / "P1.box.txt").write_text(
        "center_x=0\ncenter_y=0\ncenter_z=0\nsize_x=10\nsize_y=10\nsize_z=10\n"
    )
    ligand = tmp_path / "ligand.pdbqt"
    ligand.write_text("ligand\n")
    (shim_dir / "vina.py").write_text(
        "class Vina:\n"
        "    def __init__(self, *args, **kwargs): pass\n"
        "    def set_receptor(self, path): self.receptor = path\n"
        "    def set_ligand_from_file(self, path): self.ligand = path\n"
        "    def compute_vina_maps(self, center, box_size): self.box = (center, box_size)\n"
        "    def dock(self, exhaustiveness=8, n_poses=20): self.docked = True\n"
        "    def energies(self, n_poses=9): return [[-7.5]]\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(shim_dir)
    env["SKINSCOUT_DISABLE_LOCAL_TOOL_PATHS"] = "1"
    (tmp_path / "autodock.tsv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_autodock_run.py"),
            "--ligand-pdbqt", str(ligand),
            "--receptor-dir", str(receptor_dir),
            "--box-dir", str(box_dir),
            "--engine", "vina",
            "--out-scores", str(tmp_path / "autodock.tsv"),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode == 0, res.stderr
    assert (tmp_path / "autodock.tsv").read_text() == (
        "target_id\tvina_score\tneg_vina_score\tengine\tmap_coverage_complete\t"
        "map_coverage_numerator\tmap_coverage_denominator\tdegraded\n"
        "P1\t-7.500\t7.500\tvina\tfalse\t0\t1\ttrue\n"
    )


def test_autodock_vina_rejects_nonfinite_energy_without_output(
    tmp_path: Path,
) -> None:
    receptor_dir = tmp_path / "receptors"
    box_dir = tmp_path / "boxes"
    shim_dir = tmp_path / "shim"
    receptor_dir.mkdir()
    box_dir.mkdir()
    shim_dir.mkdir()
    (receptor_dir / "P1.pdbqt").write_text("receptor\n")
    (box_dir / "P1.box.txt").write_text(
        "center_x=0\ncenter_y=0\ncenter_z=0\nsize_x=10\nsize_y=10\nsize_z=10\n"
    )
    ligand = tmp_path / "ligand.pdbqt"
    ligand.write_text("ligand\n")
    (shim_dir / "vina.py").write_text(
        "class Vina:\n"
        "    def __init__(self, *args, **kwargs): pass\n"
        "    def set_receptor(self, path): self.receptor = path\n"
        "    def set_ligand_from_file(self, path): self.ligand = path\n"
        "    def compute_vina_maps(self, center, box_size): self.box = (center, box_size)\n"
        "    def dock(self, exhaustiveness=8, n_poses=20): self.docked = True\n"
        "    def energies(self, n_poses=9): return [[float('inf')]]\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(shim_dir)
    env["SKINSCOUT_DISABLE_LOCAL_TOOL_PATHS"] = "1"
    (tmp_path / "autodock.tsv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_autodock_run.py"),
            "--ligand-pdbqt", str(ligand),
            "--receptor-dir", str(receptor_dir),
            "--box-dir", str(box_dir),
            "--engine", "vina",
            "--out-scores", str(tmp_path / "autodock.tsv"),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "Docking energy for P1 must be finite" in res.stderr
    assert not (tmp_path / "autodock.tsv").exists()
    assert not (tmp_path / "autodock.tsv.tmp").exists()


@pytest.mark.parametrize(
    ("box_text", "message"),
    [
        (
            "center_x=nan\ncenter_y=0\ncenter_z=0\nsize_x=10\nsize_y=10\nsize_z=10\n",
            "Docking box field 'center_x' must be finite",
        ),
        (
            "center_x=0\ncenter_y=0\ncenter_z=0\nsize_x=0\nsize_y=10\nsize_z=10\n",
            "Docking box field 'size_x' must be > 0",
        ),
        (
            "center_x=0\ncenter_y=0\ncenter_z=0\nsize_y=10\nsize_z=10\n",
            "Docking box file missing required field(s) size_x",
        ),
    ],
)
def test_autodock_rejects_invalid_box_geometry_without_output(
    tmp_path: Path,
    box_text: str,
    message: str,
) -> None:
    receptor_dir = tmp_path / "receptors"
    box_dir = tmp_path / "boxes"
    receptor_dir.mkdir()
    box_dir.mkdir()
    (receptor_dir / "P1.pdbqt").write_text("receptor\n")
    (box_dir / "P1.box.txt").write_text(box_text)
    ligand = tmp_path / "ligand.pdbqt"
    ligand.write_text("ligand\n")
    env = os.environ.copy()
    env["SKINSCOUT_DISABLE_LOCAL_TOOL_PATHS"] = "1"
    (tmp_path / "autodock.tsv").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_autodock_run.py"),
            "--ligand-pdbqt", str(ligand),
            "--receptor-dir", str(receptor_dir),
            "--box-dir", str(box_dir),
            "--engine", "vina",
            "--out-scores", str(tmp_path / "autodock.tsv"),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert message in res.stderr
    assert not (tmp_path / "autodock.tsv").exists()
    assert not (tmp_path / "autodock.tsv.tmp").exists()


def test_autodock_restrict_list_preserves_rank_order_and_cap(
    tmp_path: Path,
) -> None:
    receptor_dir = tmp_path / "receptors"
    box_dir = tmp_path / "boxes"
    shim_dir = tmp_path / "shim"
    receptor_dir.mkdir()
    box_dir.mkdir()
    shim_dir.mkdir()
    for uid in ("P1", "P2", "P3"):
        (receptor_dir / f"{uid}.pdbqt").write_text(f"receptor {uid}\n")
        (box_dir / f"{uid}.box.txt").write_text(
            "center_x=0\ncenter_y=0\ncenter_z=0\nsize_x=10\nsize_y=10\nsize_z=10\n"
        )
    ligand = tmp_path / "ligand.pdbqt"
    ligand.write_text("ligand\n")
    restrict = tmp_path / "rrf.csv"
    restrict.write_text("target_id,score\nP3,0.9\nP1,0.8\nP2,0.7\n")
    (shim_dir / "vina.py").write_text(
        "class Vina:\n"
        "    def __init__(self, *args, **kwargs): pass\n"
        "    def set_receptor(self, path): self.receptor = path\n"
        "    def set_ligand_from_file(self, path): self.ligand = path\n"
        "    def compute_vina_maps(self, center, box_size): self.box = (center, box_size)\n"
        "    def dock(self, exhaustiveness=8, n_poses=20): self.docked = True\n"
        "    def energies(self, n_poses=9): return [[-6.0]]\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(shim_dir)
    env["SKINSCOUT_DISABLE_LOCAL_TOOL_PATHS"] = "1"

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_autodock_run.py"),
            "--ligand-pdbqt", str(ligand),
            "--receptor-dir", str(receptor_dir),
            "--box-dir", str(box_dir),
            "--restrict-list", str(restrict),
            "--engine", "vina",
            "--max-receptors", "2",
            "--out-scores", str(tmp_path / "autodock.tsv"),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode == 0, res.stderr
    assert (tmp_path / "autodock.tsv").read_text() == (
        "target_id\tvina_score\tneg_vina_score\tengine\tmap_coverage_complete\t"
        "map_coverage_numerator\tmap_coverage_denominator\tdegraded\n"
        "P3\t-6.000\t6.000\tvina\tfalse\t0\t2\ttrue\n"
        "P1\t-6.000\t6.000\tvina\tfalse\t0\t2\ttrue\n"
    )
