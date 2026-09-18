"""Regression tests for Stage 3 PSICHIC fail-closed behavior."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[2]


def load_stage3_psichic_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "stage3_psichic_test_module",
        ROOT / "scripts/stage3_psichic.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_input_sdf(path: Path) -> None:
    mol = Chem.MolFromSmiles("CCO")
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
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
) -> subprocess.CompletedProcess[str]:
    input_sdf = tmp_path / "ligand.sdf"
    write_input_sdf(input_sdf)
    sequence_fasta = tmp_path / "canonical.fasta"
    receptor_ids = sorted(path.stem.removesuffix("_clean") for path in clean_dir.glob("*_clean.pdb"))
    sequence_fasta.write_text(
        "".join(f">sp|{target}|TEST_HUMAN\nACDEFG\n" for target in receptor_ids)
    )

    shim_dir = tmp_path / "shim"
    shim_dir.mkdir(exist_ok=True)
    (shim_dir / "psichic.py").write_text(
        psichic_module_text or "raise ImportError('forced missing')\n"
    )

    env_pythonpath = os.environ.get("PYTHONPATH", "")
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{shim_dir}{os.pathsep}{env_pythonpath}"
        if env_pythonpath
        else str(shim_dir)
    )
    (tmp_path / "psichic.tsv").write_text("stale\n")

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
            str(tmp_path / "psichic.tsv"),
            *(extra_args or []),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_psichic_missing_fails_without_explicit_degraded_mode(tmp_path: Path) -> None:
    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    write_tiny_clean_pdb(clean_dir / "P12345_clean.pdb")

    res = run_psichic(tmp_path, clean_dir=clean_dir)

    assert res.returncode != 0
    assert "psichic is required" in res.stderr
    assert not (tmp_path / "psichic.tsv").exists()


def test_psichic_missing_degraded_mode_writes_header_only(tmp_path: Path) -> None:
    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    write_tiny_clean_pdb(clean_dir / "P12345_clean.pdb")

    res = run_psichic(tmp_path, clean_dir=clean_dir, extra_args=["--allow-unavailable"])

    assert res.returncode == 0, res.stderr
    assert (tmp_path / "psichic.tsv").read_text() == "target_id\tpsichic_score\tscore\n"


def test_psichic_fails_without_cleaned_receptors(tmp_path: Path) -> None:
    clean_dir = tmp_path / "empty_clean"
    clean_dir.mkdir()

    res = run_psichic(tmp_path, clean_dir=clean_dir, extra_args=["--allow-unavailable"])

    assert res.returncode != 0
    assert "No cleaned receptor PDBs" in res.stderr
    assert not (tmp_path / "psichic.tsv").exists()


def test_psichic_uses_canonical_fasta_not_plddt_trimmed_pdb(tmp_path: Path) -> None:
    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    write_tiny_clean_pdb(clean_dir / "P12345_clean.pdb")
    fake_psichic = """
class Model:
    def predict_batch(self, _smiles, sequences, *, score_batch_size=None):
        if sequences != ["ACDEFG"]:
            raise RuntimeError(f"unexpected sequence: {sequences}")
        return [0.5]

class PSICHIC:
    @staticmethod
    def load_pretrained(*_args, **_kwargs):
        return Model()
"""

    res = run_psichic(tmp_path, clean_dir=clean_dir, psichic_module_text=fake_psichic)

    assert res.returncode == 0, res.stderr


def test_psichic_rejects_nonfinite_model_score_without_output(
    tmp_path: Path,
) -> None:
    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    write_tiny_clean_pdb(clean_dir / "P12345_clean.pdb")

    fake_psichic = """
class Model:
    def predict_batch(self, _smiles, sequences, *, score_batch_size=None):
        return [float("nan") for _sequence in sequences]

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

    assert res.returncode != 0
    assert "PSICHIC score for P12345 must be finite" in res.stderr
    assert not (tmp_path / "psichic.tsv").exists()


def test_psichic_rejects_score_count_mismatch_without_output(
    tmp_path: Path,
) -> None:
    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    write_tiny_clean_pdb(clean_dir / "P12345_clean.pdb")

    fake_psichic = """
class Model:
    def predict_batch(self, _smiles, sequences, *, score_batch_size=None):
        return []

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

    assert res.returncode != 0
    assert "PSICHIC returned 0 score(s) for 1 receptor window(s)" in res.stderr
    assert not (tmp_path / "psichic.tsv").exists()


def test_psichic_cuda_oom_retries_smaller_batches() -> None:
    module = load_stage3_psichic_module()

    class OutOfMemoryError(RuntimeError):
        pass

    class DummyModel:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []
            self.score_batch_sizes: list[int | None] = []

        def predict_batch(
            self,
            _smiles: str,
            sequences: list[str],
            *,
            score_batch_size: int | None = None,
        ) -> list[float]:
            self.batch_sizes.append(len(sequences))
            self.score_batch_sizes.append(score_batch_size)
            if len(sequences) > 1:
                raise OutOfMemoryError("CUDA out of memory")
            return [0.5 for _ in sequences]

    model = DummyModel()
    scores = module._predict_batch_with_cuda_oom_backoff(
        model,
        "CCO",
        [("P1", "A"), ("P2", "A"), ("P3", "A")],
        7,
    )

    assert scores == [0.5, 0.5, 0.5]
    assert model.batch_sizes == [3, 1, 1, 1]
    assert model.score_batch_sizes == [7, 7, 7, 7]


def test_psichic_cuda_oom_at_single_window_uses_cpu_fallback() -> None:
    module = load_stage3_psichic_module()

    class OutOfMemoryError(RuntimeError):
        pass

    class CudaModel:
        def __init__(self) -> None:
            self.calls = 0

        def predict_batch(
            self,
            _smiles: str,
            sequences: list[str],
            *,
            score_batch_size: int | None = None,
        ) -> list[float]:
            self.calls += 1
            raise OutOfMemoryError("CUDA out of memory")

    class CpuModel:
        def __init__(self) -> None:
            self.score_batch_sizes: list[int | None] = []

        def predict_batch(
            self,
            _smiles: str,
            sequences: list[str],
            *,
            score_batch_size: int | None = None,
        ) -> list[float]:
            self.score_batch_sizes.append(score_batch_size)
            return [0.7 for _ in sequences]

    cuda_model = CudaModel()
    cpu_model = CpuModel()
    fallback_calls = 0

    def fallback() -> CpuModel:
        nonlocal fallback_calls
        fallback_calls += 1
        return cpu_model

    scores = module._predict_batch_with_cuda_oom_backoff(
        cuda_model,
        "CCO",
        [("P1", "A")],
        3,
        fallback,
    )

    assert scores == [0.7]
    assert cuda_model.calls == 1
    assert fallback_calls == 1
    assert cpu_model.score_batch_sizes == [3]


def test_psichic_sequence_windows_keep_final_long_window() -> None:
    module = load_stage3_psichic_module()

    assert module.sequence_windows("ABCDE", 5, 5) == ["ABCDE"]
    assert module.sequence_windows("ABCDEFGHIJKLM", 5, 5) == [
        "ABCDE",
        "FGHIJ",
        "IJKLM",
    ]
