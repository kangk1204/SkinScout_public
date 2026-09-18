"""Regression tests for the PSICHIC unavailable diagnostic contract (C11).

The producer must distinguish a missing prerequisite (upstream checkout,
dependency, checkpoint) from a genuine inference failure, and the disagreement
consumer must record an explicit non-claimable state instead of treating a
header-only table as an empty ranking.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pandas as pd
import pytest
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[2]


def write_input_sdf(path: Path) -> None:
    writer = Chem.SDWriter(str(path))
    writer.write(Chem.MolFromSmiles("CCO"))
    writer.close()


def write_tiny_clean_pdb(path: Path) -> None:
    path.write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\n"
        "END\n"
    )


def run_psichic(
    tmp_path: Path,
    *,
    clean_dir: Path,
    extra_args: list[str] | None = None,
    psichic_module_text: str | None = None,
    use_shim: bool = True,
    psichic_root: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    input_sdf = tmp_path / "ligand.sdf"
    write_input_sdf(input_sdf)
    sequence_fasta = tmp_path / "canonical.fasta"
    receptor_ids = sorted(
        path.stem.removesuffix("_clean") for path in clean_dir.glob("*_clean.pdb")
    )
    sequence_fasta.write_text(
        "".join(f">sp|{target}|TEST_HUMAN\nACDEFG\n" for target in receptor_ids)
    )

    env = os.environ.copy()
    if use_shim:
        shim_dir = tmp_path / "shim"
        shim_dir.mkdir(exist_ok=True)
        (shim_dir / "psichic.py").write_text(
            psichic_module_text or "raise ImportError('forced missing')\n"
        )
        env_pythonpath = os.environ.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{shim_dir}{os.pathsep}{env_pythonpath}"
            if env_pythonpath
            else str(shim_dir)
        )
    if psichic_root is not None:
        env["PSICHIC_ROOT"] = str(psichic_root)

    out_scores = tmp_path / "psichic.tsv"
    out_scores.write_text("stale\n")
    (tmp_path / "psichic.tsv.status.json").write_text("stale\n")

    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage3_psichic.py"),
            "--ligand-sdf",
            str(input_sdf),
            "--clean-dir",
            str(clean_dir),
            "--sequence-fasta",
            str(sequence_fasta),
            "--out-scores",
            str(out_scores),
            *(extra_args or []),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def read_status(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def make_clean_dir(tmp_path: Path) -> Path:
    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    write_tiny_clean_pdb(clean_dir / "P12345_clean.pdb")
    return clean_dir


PREREQUISITE_SHIM = """
class PSICHICPrerequisiteError(RuntimeError):
    def __init__(self, message, *, missing):
        super().__init__(message)
        self.missing = missing

class PSICHIC:
    @staticmethod
    def load_pretrained(*_args, **_kwargs):
        raise PSICHICPrerequisiteError("no checkpoint", missing="checkpoint:model.pt")
"""

INFERENCE_BUG_SHIM = """
class PSICHIC:
    @staticmethod
    def load_pretrained(*_args, **_kwargs):
        raise RuntimeError("inference bug")
"""


def test_prerequisite_error_with_flag_writes_header_and_status_manifest(
    tmp_path: Path,
) -> None:
    clean_dir = make_clean_dir(tmp_path)

    res = run_psichic(
        tmp_path,
        clean_dir=clean_dir,
        extra_args=["--allow-unavailable"],
        psichic_module_text=PREREQUISITE_SHIM,
    )

    assert res.returncode == 0, res.stderr
    assert (tmp_path / "psichic.tsv").read_text() == (
        "target_id\tpsichic_score\tscore\n"
    )
    status = read_status(tmp_path / "psichic.tsv.status.json")
    assert status["status"] == "unavailable"
    assert status["claimable"] is False
    assert status["reason"] == "prerequisite_missing"
    assert status["missing"] == "checkpoint:model.pt"
    assert status["targets_scored"] == 0
    assert status["schema_version"] == "skinscout.psichic-source-status.v1"


def test_prerequisite_error_without_flag_fails_closed(tmp_path: Path) -> None:
    clean_dir = make_clean_dir(tmp_path)

    res = run_psichic(
        tmp_path,
        clean_dir=clean_dir,
        psichic_module_text=PREREQUISITE_SHIM,
    )

    assert res.returncode != 0
    assert "psichic is required" in res.stderr
    assert not (tmp_path / "psichic.tsv").exists()
    assert not (tmp_path / "psichic.tsv.status.json").exists()


def test_inference_error_is_not_masked_by_allow_unavailable(tmp_path: Path) -> None:
    clean_dir = make_clean_dir(tmp_path)

    res = run_psichic(
        tmp_path,
        clean_dir=clean_dir,
        extra_args=["--allow-unavailable"],
        psichic_module_text=INFERENCE_BUG_SHIM,
    )

    assert res.returncode != 0
    assert "inference bug" in res.stderr
    assert not (tmp_path / "psichic.tsv").exists()
    assert not (tmp_path / "psichic.tsv.status.json").exists()


def test_real_adapter_missing_upstream_degrades_with_explicit_flag(
    tmp_path: Path,
) -> None:
    clean_dir = make_clean_dir(tmp_path)

    res = run_psichic(
        tmp_path,
        clean_dir=clean_dir,
        extra_args=["--allow-unavailable"],
        use_shim=False,
        psichic_root=tmp_path / "missing_psichic_root",
    )

    assert res.returncode == 0, res.stderr
    assert (tmp_path / "psichic.tsv").read_text() == (
        "target_id\tpsichic_score\tscore\n"
    )
    status = read_status(tmp_path / "psichic.tsv.status.json")
    assert status["status"] == "unavailable"
    assert status["claimable"] is False
    assert status["reason"] == "prerequisite_missing"
    assert status["missing"] == "upstream"


def test_real_adapter_missing_dependency_degrades_with_explicit_flag(
    tmp_path: Path,
) -> None:
    clean_dir = make_clean_dir(tmp_path)
    root = tmp_path / "fake_psichic_root"
    prod = root / "PSICHIC-prod"
    prod.mkdir(parents=True)
    (prod / "inference.py").write_text(
        "raise ModuleNotFoundError(\"No module named 'torch'\", name='torch')\n"
    )

    res = run_psichic(
        tmp_path,
        clean_dir=clean_dir,
        extra_args=["--allow-unavailable"],
        use_shim=False,
        psichic_root=root,
    )

    assert res.returncode == 0, res.stderr
    status = read_status(tmp_path / "psichic.tsv.status.json")
    assert status["status"] == "unavailable"
    assert status["reason"] == "prerequisite_missing"
    assert status["missing"] == "dependency:torch"


def test_real_adapter_missing_checkpoint_raises_prerequisite_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = importlib.util.spec_from_file_location(
        "psichic_adapter_checkpoint_test",
        ROOT / "psichic" / "__init__.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    root = tmp_path / "root"
    prod = root / "PSICHIC-prod"
    prod.mkdir(parents=True)
    (prod / "inference.py").write_text("")
    monkeypatch.setattr(module, "DEFAULT_ROOT", root)
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    # Keep the fake upstream import out of the shared module cache so a later
    # test cannot pick up this stub instead of the real checkout.
    monkeypatch.setitem(sys.modules, "inference", types.ModuleType("inference"))

    with pytest.raises(module.PSICHICPrerequisiteError) as excinfo:
        module.PSICHIC.load_pretrained()

    assert excinfo.value.missing == "checkpoint:config.json"


def test_real_adapter_missing_upstream_fails_without_flag(tmp_path: Path) -> None:
    clean_dir = make_clean_dir(tmp_path)

    res = run_psichic(
        tmp_path,
        clean_dir=clean_dir,
        use_shim=False,
        psichic_root=tmp_path / "missing_psichic_root",
    )

    assert res.returncode != 0
    assert "psichic is required" in res.stderr
    assert not (tmp_path / "psichic.tsv").exists()
    assert not (tmp_path / "psichic.tsv.status.json").exists()


def test_successful_run_writes_available_status_manifest(tmp_path: Path) -> None:
    clean_dir = make_clean_dir(tmp_path)
    fake_psichic = """
class Model:
    def predict_batch(self, _smiles, sequences, *, score_batch_size=None):
        return [0.5 for _sequence in sequences]

class PSICHIC:
    @staticmethod
    def load_pretrained(*_args, **_kwargs):
        return Model()
"""

    res = run_psichic(
        tmp_path,
        clean_dir=clean_dir,
        psichic_module_text=fake_psichic,
    )

    assert res.returncode == 0, res.stderr
    status = read_status(tmp_path / "psichic.tsv.status.json")
    assert status["status"] == "available"
    assert status["claimable"] is True
    assert status["targets_scored"] == 1


def write_disagreement_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    docking_top = tmp_path / "top50.csv"
    autodock = tmp_path / "autodock.tsv"
    psichic = tmp_path / "psichic_sanity.tsv"
    out = tmp_path / "disagreement.json"
    pd.DataFrame([{"target_id": "P1"}]).to_csv(docking_top, index=False)
    pd.DataFrame([{"target_id": "P1", "neg_vina_score": 9.0}]).to_csv(
        autodock, sep="\t", index=False
    )
    psichic.write_text("target_id\tpsichic_score\tscore\n")
    out.write_text("stale\n")
    return docking_top, autodock, psichic, out


def run_disagreement(
    docking_top: Path,
    autodock: Path,
    psichic: Path,
    out: Path,
    *,
    explicit_status: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts/stage3_disagreement.py"),
        "--docking-top", str(docking_top),
        "--autodock-full", str(autodock),
        "--psichic", str(psichic),
        "--chembl-dir", str(docking_top.parent / "missing_chembl"),
        "--out-json", str(out),
    ]
    if explicit_status is not None:
        command.extend(["--psichic-status", str(explicit_status)])
    return subprocess.run(
        command, capture_output=True, text=True, check=False
    )


def unavailable_status_payload() -> dict[str, object]:
    return {
        "schema_version": "skinscout.psichic-source-status.v1",
        "source": "psichic",
        "status": "unavailable",
        "claimable": False,
        "reason": "prerequisite_missing",
        "detail": "PSICHIC upstream checkout not found",
        "missing": "upstream",
        "targets_scored": 0,
    }


def test_disagreement_records_structured_unavailable_state(tmp_path: Path) -> None:
    docking_top, autodock, psichic, out = write_disagreement_inputs(tmp_path)
    psichic.with_suffix(psichic.suffix + ".status.json").write_text(
        json.dumps(unavailable_status_payload())
    )

    res = run_disagreement(docking_top, autodock, psichic, out)

    assert res.returncode == 0, res.stderr
    payload = json.loads(out.read_text())
    assert payload["status"] == "unavailable"
    assert payload["claimable"] is False
    assert payload["reason"] == "prerequisite_missing"
    assert payload["psichic_source_status"]["missing"] == "upstream"
    assert "overlap_at_top_n" not in payload


def test_disagreement_accepts_explicit_status_manifest(tmp_path: Path) -> None:
    docking_top, autodock, psichic, out = write_disagreement_inputs(tmp_path)
    status = tmp_path / "elsewhere.status.json"
    status.write_text(json.dumps(unavailable_status_payload()))

    res = run_disagreement(docking_top, autodock, psichic, out, explicit_status=status)

    assert res.returncode == 0, res.stderr
    assert json.loads(out.read_text())["status"] == "unavailable"


def test_disagreement_malformed_status_manifest_fails_closed(tmp_path: Path) -> None:
    docking_top, autodock, psichic, out = write_disagreement_inputs(tmp_path)
    psichic.with_suffix(psichic.suffix + ".status.json").write_text("{not-json\n")

    res = run_disagreement(docking_top, autodock, psichic, out)

    assert res.returncode != 0
    assert "PSICHIC source status failed to parse" in res.stderr
    assert not out.exists()


def test_disagreement_without_status_manifest_keeps_empty_frame_gate(
    tmp_path: Path,
) -> None:
    docking_top, autodock, psichic, out = write_disagreement_inputs(tmp_path)

    res = run_disagreement(docking_top, autodock, psichic, out)

    assert res.returncode != 0
    assert "PSICHIC score table contains no rows" in res.stderr
    assert not out.exists()
