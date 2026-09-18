"""Regression tests for the additive performance-v2 scientific model path."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "eval" / "performance_v2_model.py"
BUILD = ROOT / "scripts" / "build_performance_v2_embeddings.py"
TRAIN = ROOT / "scripts" / "train_performance_v2.py"
SCORE = ROOT / "scripts" / "score_performance_v2.py"


def load_model():
    spec = importlib.util.spec_from_file_location("performance_v2_model_under_test", MODEL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_build_module():
    spec = importlib.util.spec_from_file_location("performance_v2_build_under_test", BUILD)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_script(script: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def write_id_inputs(tmp_path: Path) -> tuple[Path, Path]:
    ligands = tmp_path / "ligands.csv"
    targets = tmp_path / "targets.csv"
    pd.DataFrame({"ligand_id": ["L1", "L2", "L3", "L4"]}).to_csv(ligands, index=False)
    pd.DataFrame({"target_id": ["T1", "T2", "T3"]}).to_csv(targets, index=False)
    return ligands, targets


def write_train(tmp_path: Path) -> Path:
    train = tmp_path / "train.csv"
    pd.DataFrame([
        {
            "ligand_id": "L1",
            "target_id": "T1",
            "evidence_state": "measured_positive",
            "measured_label": 1,
            "ontology": "direct_binding_reversible",
        },
        {
            "ligand_id": "L2",
            "target_id": "T1",
            "evidence_state": "measured_negative",
            "measured_label": 0,
            "ontology": "direct_binding_reversible",
        },
        {
            "ligand_id": "L2",
            "target_id": "T2",
            "evidence_state": "measured_positive",
            "measured_label": 1,
            "ontology": "functional_modulation",
        },
        {
            "ligand_id": "L3",
            "target_id": "T2",
            "evidence_state": "measured_negative",
            "measured_label": 0,
            "ontology": "functional_modulation",
        },
        {
            "ligand_id": "L4",
            "target_id": "T3",
            "evidence_state": "gray_unmeasured",
            "measured_label": "",
            "ontology": "unknown_mixed",
            "ranking_only": True,
        },
    ]).to_csv(train, index=False)
    return train


def write_prepared_input_manifest(
    module,
    tmp_path: Path,
    *,
    train: Path,
    ligands: Path,
    targets: Path,
) -> Path:
    raw = tmp_path / "raw_input_fixture.txt"
    raw.write_text("fixture\n", encoding="utf-8")
    manifest = tmp_path / "prepared_inputs.json"
    module.write_json_atomic(
        {
            "schema_version": module.INPUTS_SCHEMA,
            "inputs": {
                key: module.artifact_record(raw)
                for key in (
                    "benchmark_train",
                    "retrieval_ligands",
                    "target_fasta",
                    "target_clusters",
                )
            },
            "artifacts": {
                "train": module.artifact_record(
                    train,
                    rows=len(pd.read_csv(train)),
                ),
                "ligands": module.artifact_record(
                    ligands,
                    rows=len(pd.read_csv(ligands)),
                ),
                "targets": module.artifact_record(
                    targets,
                    rows=len(pd.read_csv(targets)),
                ),
            },
            "counts": {
                "train_rows": len(pd.read_csv(train)),
                "ligands": len(pd.read_csv(ligands)),
                "targets": len(pd.read_csv(targets)),
            },
            "schema": {
                "train_csv": module.INPUT_ARTIFACT_COLUMNS["train"],
                "ligands_csv": module.INPUT_ARTIFACT_COLUMNS["ligands"],
                "targets_csv": module.INPUT_ARTIFACT_COLUMNS["targets"],
            },
        },
        manifest,
    )
    module.validate_input_manifest(manifest)
    return manifest


def build_embeddings(tmp_path: Path) -> Path:
    ligands, targets = write_id_inputs(tmp_path)
    manifest = tmp_path / "embeddings.json"
    result = run_script(BUILD, [
        "--ligands",
        str(ligands),
        "--targets",
        str(targets),
        "--out-ligands",
        str(tmp_path / "ligand_embeddings.csv"),
        "--out-targets",
        str(tmp_path / "target_embeddings.csv"),
        "--out-manifest",
        str(manifest),
        "--fixture",
        "--fixture-dim",
        "4",
    ])
    assert result.returncode == 0, result.stderr
    return manifest


def train_model(
    tmp_path: Path,
    embedding_manifest: Path,
    *,
    train_path: Path | None = None,
) -> Path:
    manifest = tmp_path / "model.json"
    result = run_script(TRAIN, [
        "--train",
        str(train_path or write_train(tmp_path)),
        "--embedding-manifest",
        str(embedding_manifest),
        "--out-model",
        str(tmp_path / "model.npz"),
        "--out-manifest",
        str(manifest),
        "--out-budget",
        str(tmp_path / "budget.json"),
        "--fixture",
        "--seeds",
        "3,5,7",
        "--learning-rate",
        "0.01",
        "--batch-size",
        "2",
    ])
    assert result.returncode == 0, result.stderr
    return manifest


def calibration_split_evidence(*, direct_pos: int, direct_neg: int) -> dict[str, object]:
    return {
        "strategy": "sha256_ligand_target_pair_hash_target_label_set_stratified",
        "source": "train_only",
        "fraction": 0.2,
        "fit_pair_hashes_sha256": "1" * 64,
        "holdout_pair_hashes_sha256": "2" * 64,
        "fit_pairs": 2,
        "holdout_pairs": direct_pos + direct_neg,
        "fit_measured_rows": 2,
        "holdout_measured_rows": direct_pos + direct_neg,
        "disjoint_pair_hashes": True,
        "event_counts": {
            "direct_binding_reversible": {
                "fit_pos": 1,
                "fit_neg": 1,
                "holdout_pos": direct_pos,
                "holdout_neg": direct_neg,
            },
            "functional_modulation": {
                "fit_pos": 0,
                "fit_neg": 0,
                "holdout_pos": 0,
                "holdout_neg": 0,
            },
        },
    }


def write_torch_precomputed_inputs(tmp_path: Path) -> tuple[Path, Path]:
    module = load_model()
    ligand_ids = [f"L{index}" for index in range(1, 9)]
    target_ids = ["T1", "T2", "T3"]
    ligand_dir = tmp_path / "torch_ligands"
    target_dir = tmp_path / "torch_targets"
    ligand_dir.mkdir()
    target_dir.mkdir()
    ligand_shard = ligand_dir / "ligands_000000.npz"
    target_shard = target_dir / "targets_000000.npz"
    ligand_vectors = np.vstack([
        module.deterministic_vector(f"ligand:{identifier}", 6) for identifier in ligand_ids
    ]).astype(np.float32)
    target_vectors = np.vstack([
        module.deterministic_vector(f"target:{identifier}", 4) for identifier in target_ids
    ]).astype(np.float32)
    np.savez_compressed(ligand_shard, ids=np.array(ligand_ids), vectors=ligand_vectors)
    np.savez_compressed(target_shard, ids=np.array(target_ids), vectors=target_vectors)
    embedding_manifest = tmp_path / "torch_embeddings.json"
    module.write_json_atomic({
        "schema_version": module.EMBEDDING_SCHEMA,
        "mode": "fixture_precomputed",
        "checkpoints": {
            "ligand": {
                "model_id": module.MOLFORMER_CHECKPOINT,
                "revision": module.MOLFORMER_REVISION,
            },
            "protein": {
                "model_id": module.ESM2_CHECKPOINT,
                "revision": module.ESM2_REVISION,
            },
        },
        "artifacts": {
            "ligands": {
                "format": "npz_shards",
                "path": str(ligand_dir),
                "dtype": "float32",
                "dim": 6,
                "rows": len(ligand_ids),
                "shards": [module.artifact_record(ligand_shard, rows=len(ligand_ids))],
            },
            "targets": {
                "format": "npz_shards",
                "path": str(target_dir),
                "dtype": "float32",
                "dim": 4,
                "rows": len(target_ids),
                "shards": [module.artifact_record(target_shard, rows=len(target_ids))],
            },
        },
        "inputs": {
            "ligands": module.artifact_record(ligand_shard),
            "targets": module.artifact_record(target_shard),
        },
        "provenance": {
            "deterministic_canonical_ids": True,
            "source": "download_free_torch_runtime_fixture",
        },
        "freeze_contract": {
            "encoders_frozen": True,
            "requires_grad": False,
            "no_grad": True,
            "fixture": True,
        },
        "license_profile": {"redistribution": "test"},
    }, embedding_manifest)

    rows = []
    for ligand_id, label in (("L1", 1), ("L2", 1), ("L3", 0), ("L4", 0)):
        rows.append({
            "ligand_id": ligand_id,
            "target_id": "T1",
            "evidence_state": "measured_positive" if label else "measured_negative",
            "measured_label": label,
            "ontology": "direct_binding_reversible",
        })
    for ligand_id, label in (("L5", 1), ("L6", 1), ("L7", 0), ("L8", 0)):
        rows.append({
            "ligand_id": ligand_id,
            "target_id": "T2",
            "evidence_state": "measured_positive" if label else "measured_negative",
            "measured_label": label,
            "ontology": "functional_modulation",
        })
    for ligand_id in ("L1", "L2", "L5", "L6"):
        rows.append({
            "ligand_id": ligand_id,
            "target_id": "T3",
            "evidence_state": "gray_unmeasured",
            "measured_label": "",
            "ontology": "unknown_mixed",
            "ranking_only": True,
        })
    train_csv = tmp_path / "torch_train.csv"
    pd.DataFrame(rows).to_csv(train_csv, index=False)
    return embedding_manifest, train_csv


def test_pure_contract_import_without_heavy_ml_modules() -> None:
    module = load_model()
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules
    assert module.ONTOLOGY == (
        "direct_binding_reversible",
        "functional_modulation",
        "cofactor_substrate",
        "reactive_covalent_sensor",
        "metabolite_prodrug",
        "pathway_effect",
        "unknown_mixed",
    )


def test_ranking_columns_are_exact_and_unique() -> None:
    module = load_model()
    assert module.RANKING_COLUMNS == (
        "ligand_id",
        "target_id",
        "performance_v2_ranking_score",
        "performance_v2_score_semantics",
        "performance_v2_ood_route",
        "performance_v2_abstained",
        "performance_v2_direct_binding_probability",
        "performance_v2_functional_modulation_probability",
        "performance_v2_rank",
    )
    assert len(module.RANKING_COLUMNS) == len(set(module.RANKING_COLUMNS))


def test_transformer_loader_stays_fp32_until_device_move_and_forwards_kwargs() -> None:
    module = load_model()
    calls: list[tuple[str, object]] = []

    class FakeModel:
        def to(self, **kwargs):
            calls.append(("to", kwargs))
            return self

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append(("load", {"model_id": model_id, **kwargs}))
            assert "torch_dtype" not in kwargs
            return FakeModel()

    device = SimpleNamespace(type="cuda")
    module.load_transformer_fp32_then_place(
        FakeAutoModel,
        model_id="fixture/model",
        revision="a" * 40,
        trust_remote_code=True,
        device=device,
        model_kwargs={"add_pooling_layer": False},
    )
    assert calls == [
        (
            "load",
            {
                "model_id": "fixture/model",
                "revision": "a" * 40,
                "trust_remote_code": True,
                "add_pooling_layer": False,
            },
        ),
        ("to", {"device": device}),
    ]


def test_protein_windows_cover_full_sequence_with_fixed_overlap_and_no_truncation() -> None:
    build = load_build_module()
    sequence = "A" * 2620
    windows = build._protein_windows(sequence, residue_window=1022, overlap=128)
    assert [(start, stop) for start, stop, _ in windows] == [
        (0, 1022),
        (894, 1916),
        (1788, 2620),
    ]
    assert all(windows[index][1] - windows[index + 1][0] == 128 for index in range(2))
    covered = np.zeros(len(sequence), dtype=bool)
    for start, stop, window in windows:
        assert window == sequence[start:stop]
        assert len(window) <= 1022
        covered[start:stop] = True
    assert covered.all()

    class FakeTokenizer:
        def __init__(self) -> None:
            self.kwargs = None

        def __call__(self, texts, **kwargs):
            self.kwargs = kwargs
            width = max(map(len, texts)) + 2
            return {
                "input_ids": np.zeros((len(texts), width), dtype=np.int64),
                "special_tokens_mask": np.zeros((len(texts), width), dtype=np.int64),
            }

    tokenizer = FakeTokenizer()
    build._tokenize_protein_windows(
        tokenizer, [window for _, _, window in windows], max_length=1024
    )
    assert tokenizer.kwargs["truncation"] is False
    assert tokenizer.kwargs["return_special_tokens_mask"] is True


def test_esm_pooling_ignores_random_pooler_and_special_tokens() -> None:
    torch = pytest.importorskip("torch")
    build = load_build_module()
    hidden = torch.tensor([[[100.0], [1.0], [3.0], [200.0], [0.0]]])
    outputs = SimpleNamespace(
        pooler_output=torch.tensor([[999.0]]),
        last_hidden_state=hidden,
    )
    attention = torch.tensor([[1, 1, 1, 1, 0]])
    special = torch.tensor([[1, 0, 0, 1, 1]])
    pooled = build._residue_pooled_output(outputs, attention, special)
    assert pooled.item() == pytest.approx(2.0)


def test_checkpoint_embedding_requires_cuda_without_explicit_cpu() -> None:
    build = load_build_module()

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeTorch:
        cuda = FakeCuda()

        @staticmethod
        def device(value):
            return SimpleNamespace(type=value)

    with pytest.raises(SystemExit, match="required unless --cpu"):
        build._embedding_device(FakeTorch, cpu=False)
    assert build._embedding_device(FakeTorch, cpu=True).type == "cpu"
    parsed = build.parse_args([
        "--ligands", "ligands.csv",
        "--targets", "targets.csv",
        "--out-ligands", "ligands-out",
        "--out-targets", "targets-out",
        "--out-manifest", "manifest.json",
    ])
    assert parsed.protein_batch_size == 8


def test_pu_separation_never_treats_gray_as_negative() -> None:
    module = load_model()
    frame = pd.DataFrame([
        {"target_id": "T1", "evidence_state": "measured_positive", "measured_label": 1},
        {"target_id": "T1", "evidence_state": "measured_negative", "measured_label": 0},
        {
            "target_id": "T2",
            "evidence_state": "gray_unmeasured",
            "measured_label": "",
            "ranking_only": True,
        },
    ])
    measured, ranking_only = module.split_pu_training(frame)
    assert measured["measured_label"].tolist() == [1, 0]
    assert ranking_only["target_id"].tolist() == ["T2"]
    bad = frame.copy()
    bad.loc[2, "measured_label"] = 0
    with pytest.raises(module.ContractError, match="never be labelled negative"):
        module.split_pu_training(bad)


def test_target_balancing_and_effective_target_count() -> None:
    module = load_model()
    frame = pd.DataFrame({
        "target_id": ["T1", "T1", "T1", "T2"],
        "measured_label": [1, 1, 0, 0],
    })
    weights = module.target_balanced_weights(frame)
    assert pytest.approx(weights.sum()) == len(frame)
    assert weights[0] == pytest.approx(weights[1])
    assert weights[2] > weights[0]
    assert module.effective_target_count(frame, weights) == pytest.approx(2.0)


def test_effective_target_count_ignores_zero_weight_targets() -> None:
    """A zero-weight target must drop out, not poison the entropy with NaN."""
    module = load_model()
    frame = pd.DataFrame({"target_id": ["T1", "T2"], "measured_label": [1, 0]})

    assert module.effective_target_count(frame, [1.0, 0.0]) == pytest.approx(1.0)
    assert module.effective_target_count(frame, [1.0, 1.0]) == pytest.approx(2.0)
    assert module.effective_target_count(frame, [0.0, 0.0]) == 0.0


def test_effective_target_count_rejects_a_mismatched_weight_vector() -> None:
    module = load_model()
    frame = pd.DataFrame({"target_id": ["T1", "T2"], "measured_label": [1, 0]})

    with pytest.raises(ValueError):
        module.effective_target_count(frame, [1.0])


def test_train_only_prior_beta_validation_and_debias() -> None:
    module = load_model()
    correction = module.prior_correction_beta(
        train_pos=8,
        train_neg=2,
        deploy_pos=2,
        deploy_neg=8,
        beta=0.5,
        source="train",
    )
    assert correction < 0
    with pytest.raises(module.ContractError, match="train-only"):
        module.prior_correction_beta(
            train_pos=8,
            train_neg=2,
            deploy_pos=2,
            deploy_neg=8,
            beta=0.5,
            source="dev",
        )
    with pytest.raises(module.ContractError, match=r"\[0, 1\]"):
        module.prior_correction_beta(
            train_pos=8,
            train_neg=2,
            deploy_pos=2,
            deploy_neg=8,
            beta=1.1,
        )


def test_calibration_semantics_require_measured_coverage() -> None:
    module = load_model()
    assert module.fit_platt([0.2, 0.3], [1, 1]) is None
    cal = module.fit_platt([-1.0, 1.0, -0.5, 0.8], [0, 1, 0, 1])
    assert cal is not None
    probs = cal.predict([-2.0, 2.0])
    assert 0.0 < probs[0] < probs[1] < 1.0
    manifest = module.calibration_manifest(
        cal,
        None,
        split_evidence=calibration_split_evidence(direct_pos=2, direct_neg=2),
        model_state_sha256="a" * 64,
    )
    assert manifest["direct_binding_reversible"]["semantics"] == "measured_event_probability"
    assert manifest["functional_modulation"]["semantics"] == "not_a_probability"


def test_calibration_holdout_is_pair_hash_stratified_and_disjoint() -> None:
    module = load_model()
    rows = []
    for ontology, target, prefix in (
        ("direct_binding_reversible", "T1", "D"),
        ("functional_modulation", "T2", "F"),
    ):
        for label, state in ((1, "measured_positive"), (0, "measured_negative")):
            for offset in range(2):
                rows.append({
                    "ligand_id": f"{prefix}{label}{offset}",
                    "target_id": target,
                    "evidence_state": state,
                    "measured_label": label,
                    "ontology": ontology,
                })
    measured = pd.DataFrame(rows)
    fit, holdout, evidence = module.calibration_holdout_split(measured)
    fit_pairs = set(zip(fit["ligand_id"], fit["target_id"], fit["ontology"], strict=True))
    holdout_pairs = set(zip(holdout["ligand_id"], holdout["target_id"], holdout["ontology"], strict=True))
    assert fit_pairs.isdisjoint(holdout_pairs)
    assert len(fit) + len(holdout) == len(measured)
    assert evidence["holdout_measured_rows"] == 4
    shuffled = measured.sample(frac=1.0, random_state=9).reset_index(drop=True)
    _, _, shuffled_evidence = module.calibration_holdout_split(shuffled)
    assert shuffled_evidence == evidence


def test_ecfp_preserves_all_morgan_bits_by_identity() -> None:
    module = load_model()
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    bits = module.ecfp_bit_vector("CCO")
    expected = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048
    ).GetFingerprint(Chem.MolFromSmiles("CCO"))
    assert bits.shape == (2048,)
    assert bits.dtype == np.float32
    assert set(np.unique(bits)) <= {0.0, 1.0}
    assert np.flatnonzero(bits).tolist() == list(expected.GetOnBits())


def test_ood_routes_and_abstention() -> None:
    module = load_model()
    assert module.ood_route("L1", "T1", ["L1"], ["T1"]) == "in_domain"
    assert module.ood_route("L2", "T1", ["L1"], ["T1"]) == "ligand_cold"
    assert module.ood_route("L1", "T2", ["L1"], ["T1"]) == "target_cold"
    assert module.ood_route("L2", "T2", ["L1"], ["T1"]) == "dual_cold"
    assert module.ood_route("L1", "T1", ["L1"], ["T1"], structure_abstained=True) == "structure_abstained"
    assert module.should_abstain("dual_cold")
    assert module.should_abstain("structure_abstained")
    assert module.should_abstain("in_domain", score=0.1, min_score=0.5)
    assert not module.should_abstain("ligand_cold", score=0.6, min_score=0.5)


def test_tamper_hash_rejection(tmp_path: Path) -> None:
    module = load_model()
    manifest = build_embeddings(tmp_path)
    payload = json.loads(manifest.read_text())
    ligand_path = Path(payload["artifacts"]["ligands"]["path"])
    ligand_path.write_text(ligand_path.read_text() + "\n")
    with pytest.raises(module.ContractError, match="SHA-256 mismatch"):
        module.validate_embedding_manifest(manifest)


def test_budget_rejection(tmp_path: Path) -> None:
    module = load_model()
    budget = tmp_path / "budget.json"
    module.write_json_atomic({
        "schema_version": module.BUDGET_SCHEMA,
        "execution_mode": "cuda",
        "peak_vram_gib": 11.6,
        "peak_vram_source": "torch.cuda.max_memory_allocated",
        "peak_vram_measured": True,
        "trainable_params": 5,
        "fallback_model_ids": ["fixture"],
        "license_profile": {"redistribution": "ok"},
    }, budget)
    with pytest.raises(module.ContractError, match="peak_vram_gib exceeds"):
        module.validate_budget_manifest(budget)
    module.write_json_atomic({
        "schema_version": module.BUDGET_SCHEMA,
        "execution_mode": "cpu",
        "peak_vram_gib": 1.0,
        "peak_vram_source": "not_applicable_cpu",
        "peak_vram_measured": False,
        "trainable_params": 10_000_001,
        "fallback_model_ids": ["fixture"],
        "license_profile": {"redistribution": "ok"},
    }, budget)
    with pytest.raises(module.ContractError, match="trainable_params exceeds"):
        module.validate_budget_manifest(budget)
    module.write_json_atomic({
        "schema_version": module.BUDGET_SCHEMA,
        "execution_mode": "cuda",
        "peak_vram_gib": 0.0,
        "peak_vram_source": "caller_supplied",
        "peak_vram_measured": False,
        "trainable_params": 5,
        "fallback_model_ids": ["fixture"],
        "license_profile": {"redistribution": "ok"},
    }, budget)
    with pytest.raises(module.ContractError, match="measured torch.cuda"):
        module.validate_budget_manifest(budget)


def test_deterministic_fixture_train_and_score(tmp_path: Path) -> None:
    embedding_manifest = build_embeddings(tmp_path)
    model_manifest = train_model(tmp_path, embedding_manifest)
    trained = json.loads(model_manifest.read_text())
    assert trained["training_hyperparameters"]["requested_batch_size"] == 2
    assert trained["training_hyperparameters"]["learning_rate"] == pytest.approx(0.01)
    assert trained["calibration"]["logit_source"] == "saved_parameter_average"
    budget = json.loads(Path(trained["inputs"]["budget_manifest"]["path"]).read_text())
    assert budget["execution_mode"] == "fixture"
    assert budget["peak_vram_source"] == "not_applicable_cpu"
    out_ranking = tmp_path / "ranking.csv"
    out_manifest = tmp_path / "ranking.json"
    args = [
        "--model-manifest",
        str(model_manifest),
        "--ligand-id",
        "L1",
        "--out-ranking",
        str(out_ranking),
        "--out-manifest",
        str(out_manifest),
    ]
    first = run_script(SCORE, args)
    assert first.returncode == 0, first.stderr
    first_csv = out_ranking.read_text()
    second = run_script(SCORE, args)
    assert second.returncode != 0
    assert "refusing to overwrite pre-existing output" in second.stderr
    assert out_ranking.read_text() == first_csv
    ranking = pd.read_csv(out_ranking)
    assert list(ranking["performance_v2_score_semantics"].unique()) == [
        "ranking_score_not_probability"
    ]
    assert set(ranking["performance_v2_ood_route"]) <= {
        "in_domain",
        "ligand_cold",
        "target_cold",
        "dual_cold",
        "structure_abstained",
    }
    payload = json.loads(out_manifest.read_text())
    assert payload["semantics"]["ranking_score"] == "not_probability"
    module = load_model()
    module.validate_ranking_manifest(out_manifest)


def test_model_manifest_rejects_rehashed_train_domain_tamper(tmp_path: Path) -> None:
    module = load_model()
    embedding_manifest = build_embeddings(tmp_path)
    model_manifest = train_model(tmp_path, embedding_manifest)
    payload = json.loads(model_manifest.read_text(encoding="utf-8"))
    payload["train_targets"] = [*payload["train_targets"], "T_HELD_OUT"]
    payload["train_domain_sha256"] = module.sha256_payload(
        {
            "train_ligands": payload["train_ligands"],
            "train_targets": payload["train_targets"],
        }
    )
    module.write_json_atomic(payload, model_manifest)

    with pytest.raises(module.ContractError, match="train_targets does not match"):
        module.validate_model_manifest(model_manifest)


def test_smiles_and_compound_json_fixture_scoring_record_query_provenance(tmp_path: Path) -> None:
    embedding_manifest = build_embeddings(tmp_path)
    model_manifest = train_model(tmp_path, embedding_manifest)
    smiles_ranking = tmp_path / "smiles_ranking.csv"
    smiles_manifest = tmp_path / "smiles_ranking.json"
    result = run_script(SCORE, [
        "--model-manifest",
        str(model_manifest),
        "--smiles",
        "CCO",
        "--out-ranking",
        str(smiles_ranking),
        "--out-manifest",
        str(smiles_manifest),
    ])
    assert result.returncode == 0, result.stderr
    payload = json.loads(smiles_manifest.read_text())
    assert payload["query"]["input_type"] == "smiles"
    assert len(payload["query"]["canonical_smiles_sha256"]) == 64
    assert payload["query"]["raw_smiles_recorded"] is False
    assert "CCO" not in json.dumps(payload)

    compound_json = tmp_path / "compound_canonical.json"
    compound_json.write_text(json.dumps({
        "input_type": "smiles",
        "input_smiles": "OCC",
        "input_canonical_smiles": "CCO",
        "canonical_smiles": "CCO",
    }) + "\n")
    compound_manifest = tmp_path / "compound_ranking.json"
    result = run_script(SCORE, [
        "--model-manifest",
        str(model_manifest),
        "--compound-json",
        str(compound_json),
        "--out-ranking",
        str(tmp_path / "compound_ranking.csv"),
        "--out-manifest",
        str(compound_manifest),
    ])
    assert result.returncode == 0, result.stderr
    payload = json.loads(compound_manifest.read_text())
    assert payload["query"]["compound_json_smiles_field"] == "canonical_smiles"
    assert payload["query"]["compound_json"]["sha256"]


def test_score_query_options_are_mutually_exclusive(tmp_path: Path) -> None:
    embedding_manifest = build_embeddings(tmp_path)
    model_manifest = train_model(tmp_path, embedding_manifest)
    result = run_script(SCORE, [
        "--model-manifest",
        str(model_manifest),
        "--ligand-id",
        "L1",
        "--smiles",
        "CCO",
        "--out-ranking",
        str(tmp_path / "ranking.csv"),
        "--out-manifest",
        str(tmp_path / "ranking.json"),
    ])
    assert result.returncode != 0
    assert "not allowed with argument" in result.stderr


def test_sharded_npz_embedding_manifest_roundtrip_without_downloads(tmp_path: Path) -> None:
    module = load_model()
    ligand_dir = tmp_path / "ligand_shards"
    target_dir = tmp_path / "target_shards"
    ligand_dir.mkdir()
    target_dir.mkdir()
    ligand_shard = ligand_dir / "ligands_000000.npz"
    target_shard = target_dir / "targets_000000.npz"
    np_lig = __import__("numpy")
    np_lig.savez_compressed(
        ligand_shard,
        ids=np_lig.array(["L1", "L2"], dtype="U"),
        vectors=np_lig.array([[1, 2, 3], [4, 5, 6]], dtype=np_lig.float32),
    )
    np_lig.savez_compressed(
        target_shard,
        ids=np_lig.array(["T1"], dtype="U"),
        vectors=np_lig.array([[0.1, 0.2]], dtype=np_lig.float32),
    )
    manifest = tmp_path / "embeddings.json"
    module.write_json_atomic({
        "schema_version": module.EMBEDDING_SCHEMA,
        "mode": "fixture_precomputed",
        "checkpoints": {
            "ligand": {
                "model_id": module.MOLFORMER_CHECKPOINT,
                "revision": module.MOLFORMER_REVISION,
            },
            "protein": {
                "model_id": module.ESM2_CHECKPOINT,
                "revision": module.ESM2_REVISION,
            },
        },
        "artifacts": {
            "ligands": {
                "format": "npz_shards",
                "path": str(ligand_dir),
                "dtype": "float32",
                "dim": 3,
                "rows": 2,
                "shards": [module.artifact_record(ligand_shard, rows=2)],
            },
            "targets": {
                "format": "npz_shards",
                "path": str(target_dir),
                "dtype": "float32",
                "dim": 2,
                "rows": 1,
                "shards": [module.artifact_record(target_shard, rows=1)],
            },
        },
        "inputs": {
            "ligands": module.artifact_record(ligand_shard),
            "targets": module.artifact_record(target_shard),
        },
        "provenance": {
            "deterministic_canonical_ids": True,
            "streaming": "test shard",
        },
        "freeze_contract": {
            "encoders_frozen": True,
            "requires_grad": False,
            "no_grad": True,
        },
        "license_profile": {"redistribution": "ok"},
    }, manifest)
    module.validate_embedding_manifest(manifest)
    ids, matrix = module.load_embedding_artifact(
        json.loads(manifest.read_text())["artifacts"]["ligands"],
        base_dir=tmp_path,
        id_column="ligand_id",
        label="ligands",
    )
    assert ids == ["L1", "L2"]
    assert matrix.shape == (2, 3)


def test_shard_local_batches_avoid_ligand_thrash_and_target_cache_is_bounded(
    tmp_path: Path,
) -> None:
    module = load_model()
    shard_records = []
    for shard_index in range(5):
        path = tmp_path / f"shard_{shard_index}.npz"
        ids = np.array([f"L{shard_index}_0", f"L{shard_index}_1"])
        vectors = np.full((2, 3), shard_index, dtype=np.float32)
        np.savez_compressed(path, ids=ids, vectors=vectors)
        shard_records.append(module.artifact_record(path, rows=2))
    record = {
        "format": "npz_shards",
        "path": str(tmp_path),
        "dtype": "float32",
        "dim": 3,
        "rows": 10,
        "shards": shard_records,
    }
    store = module.EmbeddingStore(
        record,
        base_dir=tmp_path,
        id_column="ligand_id",
        label="locality fixture",
        cache_shards=1,
    )
    embedding_indexes = np.array([0, 2, 4, 6, 8, 1, 3, 5, 7, 9])
    locality = np.array([store.locality_key(int(index)) for index in embedding_indexes])
    first_sampler = module.DeterministicShardBatchSampler(locality, batch_size=2, seed=17)
    first_batches = list(first_sampler)
    second_batches = list(
        module.DeterministicShardBatchSampler(locality, batch_size=2, seed=17)
    )
    assert first_batches == second_batches
    for row_offsets in first_batches:
        batch_indexes = embedding_indexes[row_offsets]
        assert len({store.locality_key(int(index)) for index in batch_indexes}) == 1
        store.fetch(batch_indexes)
    assert sorted(offset for batch in first_batches for offset in batch) == list(range(10))
    assert store.shard_cache_misses == 5

    target_store = module.EmbeddingStore(
        record,
        base_dir=tmp_path,
        id_column="target_id",
        label="target cache fixture",
        cache_shards=8,
    )
    for _ in range(3):
        target_store.fetch(np.array([8, 0, 6, 2, 4]))
    assert target_store.shard_cache_misses == 5


def test_ood_route_reuses_precomputed_set_domains() -> None:
    module = load_model()

    class MembershipOnlySet(frozenset):
        def __iter__(self):
            raise AssertionError("set domain was rebuilt by iteration")

    ligand_domain = MembershipOnlySet({"L1"})
    target_domain = MembershipOnlySet({"T1", "T2"})
    assert module.ood_route("L1", "T2", ligand_domain, target_domain) == "in_domain"
    assert module.ood_route("L2", "T2", ligand_domain, target_domain) == "ligand_cold"


def test_production_embedding_manifest_requires_commits_and_binary_ecfp(tmp_path: Path) -> None:
    module = load_model()
    ligand_dir = tmp_path / "ligands"
    target_dir = tmp_path / "targets"
    ligand_dir.mkdir()
    target_dir.mkdir()
    ligand_shard = ligand_dir / "ligands_000000.npz"
    target_shard = target_dir / "targets_000000.npz"
    ecfp = module.ecfp_bit_vector("CCO")
    ligand_vector = np.concatenate([np.array([0.1, -0.2], dtype=np.float32), ecfp])[None, :]
    np.savez_compressed(ligand_shard, ids=np.array(["L1"]), vectors=ligand_vector)
    np.savez_compressed(
        target_shard,
        ids=np.array(["T1"]),
        vectors=np.array([[0.3, 0.4, 0.5]], dtype=np.float32),
    )
    source_ligands = tmp_path / "source_ligands.csv"
    source_targets = tmp_path / "source_targets.csv"
    source_train = tmp_path / "source_train.csv"
    pd.DataFrame({"ligand_id": ["L1"], "smiles": ["CCO"]}).to_csv(
        source_ligands,
        index=False,
    )
    pd.DataFrame(
        {
            "target_id": ["T1"],
            "sequence": ["ACD"],
            "target_cluster_30": ["C30"],
            "target_cluster_50": ["C50"],
        }
    ).to_csv(source_targets, index=False)
    pd.DataFrame(
        {
            "ligand_id": ["L1"],
            "target_id": ["T1"],
            "evidence_state": ["measured_positive"],
            "measured_label": [1],
            "ontology": ["direct_binding_reversible"],
        }
    ).to_csv(source_train, index=False)
    input_manifest = write_prepared_input_manifest(
        module,
        tmp_path,
        train=source_train,
        ligands=source_ligands,
        targets=source_targets,
    )
    manifest = tmp_path / "production_embeddings.json"
    payload = {
        "schema_version": module.EMBEDDING_SCHEMA,
        "mode": "frozen_checkpoint",
        "checkpoints": {
            "ligand": {
                "model_id": module.MOLFORMER_CHECKPOINT,
                "revision": module.MOLFORMER_REVISION,
                "resolved_revision": "a" * 40,
                "license_id": module.MOLFORMER_LICENSE,
            },
            "protein": {
                "model_id": module.ESM2_CHECKPOINT,
                "hf_model_id": f"facebook/{module.ESM2_CHECKPOINT}",
                "revision": module.ESM2_REVISION,
                "resolved_revision": "b" * 40,
                "license_id": module.ESM2_LICENSE,
            },
        },
        "artifacts": {
            "ligands": {
                "format": "npz_shards",
                "path": str(ligand_dir),
                "dtype": "float32",
                "dim": 2050,
                "rows": 1,
                "molformer_dim": 2,
                "ecfp_offset": 2,
                "ecfp_bits": 2048,
                "ecfp_radius": 2,
                "ecfp_encoding": "independent_binary_float32",
                "shards": [module.artifact_record(ligand_shard, rows=1)],
            },
            "targets": {
                "format": "npz_shards",
                "path": str(target_dir),
                "dtype": "float32",
                "dim": 3,
                "rows": 1,
                "shards": [module.artifact_record(target_shard, rows=1)],
            },
        },
        "inputs": {
            "ligands": module.artifact_record(source_ligands),
            "targets": module.artifact_record(source_targets),
            "input_manifest": module.artifact_record(input_manifest),
        },
        "provenance": {
            "deterministic_canonical_ids": True,
            "precision": {
                "ligand": module.MOLFORMER_PRECISION,
                "protein": "fp32_model_weights_with_forward_autocast_fp16",
            },
            "ligand_tokenization": {
                "strategy": "tokenizer_truncation_fixed_max_tokens",
                "max_tokens": 256,
                "truncation": True,
            },
            "pooling": {
                "ligand": module.MOLFORMER_POOLING,
                "protein": module.PROTEIN_POOLING,
                "protein_add_pooling_layer": False,
            },
            "protein_windowing": {
                "strategy": module.PROTEIN_WINDOW_STRATEGY,
                "max_tokens": 1024,
                "special_tokens_per_window": 2,
                "residue_window": 1022,
                "overlap_residues": 128,
                "sequence_count": 1,
                "long_sequence_count": 0,
                "total_window_count": 1,
                "full_sequence_coverage": True,
                "silent_truncation": False,
            },
        },
        "freeze_contract": {
            "encoders_frozen": True,
            "requires_grad": False,
            "no_grad": True,
        },
        "license_profile": {"redistribution": "ok"},
    }
    module.write_json_atomic(payload, manifest)
    module.validate_embedding_manifest(manifest)

    payload["provenance"]["precision"]["ligand"] = (
        "fp32_model_weights_with_forward_autocast_fp16"
    )
    module.write_json_atomic(payload, manifest)
    with pytest.raises(module.ContractError, match="MoLFormer must use fp32"):
        module.validate_embedding_manifest(manifest)

    payload["provenance"]["precision"]["ligand"] = module.MOLFORMER_PRECISION
    payload["checkpoints"]["ligand"].pop("resolved_revision")
    module.write_json_atomic(payload, manifest)
    with pytest.raises(module.ContractError, match="immutable 40-hex"):
        module.validate_embedding_manifest(manifest)

    payload["checkpoints"]["ligand"]["resolved_revision"] = "a" * 40
    ligand_vector[0, 2] = 0.5
    np.savez_compressed(ligand_shard, ids=np.array(["L1"]), vectors=ligand_vector)
    payload["artifacts"]["ligands"]["shards"] = [
        module.artifact_record(ligand_shard, rows=1)
    ]
    module.write_json_atomic(payload, manifest)
    with pytest.raises(module.ContractError, match="independent binary bits"):
        module.validate_embedding_manifest(manifest)

    ligand_vector[0, 2] = 0.0
    ligand_vector[0, 0] = np.nan
    np.savez_compressed(ligand_shard, ids=np.array(["L1"]), vectors=ligand_vector)
    payload["artifacts"]["ligands"]["shards"] = [
        module.artifact_record(ligand_shard, rows=1)
    ]
    module.write_json_atomic(payload, manifest)
    with pytest.raises(module.ContractError, match="all be finite"):
        module.validate_embedding_manifest(manifest)


def test_embedding_shard_writer_rejects_nonfinite_vectors(tmp_path: Path) -> None:
    build = load_build_module()
    with pytest.raises(SystemExit, match="nonfinite embedding shard"):
        build._write_npz_shard(
            tmp_path / "bad.npz",
            ["L1"],
            np.array([[np.inf]], dtype=np.float32),
        )
    assert not (tmp_path / "bad.npz").exists()


def test_embedding_cli_validates_separate_protein_batch_size(tmp_path: Path) -> None:
    ligands, targets = write_id_inputs(tmp_path)
    result = run_script(BUILD, [
        "--ligands", str(ligands),
        "--targets", str(targets),
        "--out-ligands", str(tmp_path / "ligand.csv"),
        "--out-targets", str(tmp_path / "target.csv"),
        "--out-manifest", str(tmp_path / "embedding.json"),
        "--fixture",
        "--protein-batch-size", "0",
    ])
    assert result.returncode != 0
    assert "--protein-batch-size must be >= 1" in result.stderr


def test_prepared_input_manifest_hash_chain_reaches_model(tmp_path: Path) -> None:
    module = load_model()
    ligands, targets = write_id_inputs(tmp_path)
    pd.DataFrame(
        {
            "ligand_id": ["L1", "L2", "L3", "L4"],
            "smiles": ["CCO", "CCN", "CCC", "CCCl"],
        }
    ).to_csv(ligands, index=False)
    pd.DataFrame(
        {
            "target_id": ["T1", "T2", "T3"],
            "sequence": ["ACD", "EFG", "HIK"],
            "target_cluster_30": ["C30A", "C30B", "C30C"],
            "target_cluster_50": ["C50A", "C50B", "C50C"],
        }
    ).to_csv(targets, index=False)
    train = write_train(tmp_path)
    train_frame = pd.read_csv(train)
    train_frame.drop(columns=["ranking_only"]).to_csv(train, index=False)
    input_manifest = write_prepared_input_manifest(
        module,
        tmp_path,
        train=train,
        ligands=ligands,
        targets=targets,
    )
    embedding_manifest = tmp_path / "lineage_embeddings.json"
    built = run_script(
        BUILD,
        [
            "--ligands",
            str(ligands),
            "--targets",
            str(targets),
            "--input-manifest",
            str(input_manifest),
            "--out-ligands",
            str(tmp_path / "lineage_ligands.csv"),
            "--out-targets",
            str(tmp_path / "lineage_targets.csv"),
            "--out-manifest",
            str(embedding_manifest),
            "--fixture",
            "--fixture-dim",
            "4",
        ],
    )
    assert built.returncode == 0, built.stderr
    embedding_payload = module.validate_embedding_manifest(embedding_manifest)
    assert embedding_payload["inputs"]["input_manifest"]["sha256"] == module.sha256_file(
        input_manifest
    )

    model_manifest = train_model(
        tmp_path,
        embedding_manifest,
        train_path=train,
    )
    model_payload = module.validate_model_manifest(model_manifest)
    assert model_payload["inputs"]["embedding_manifest"]["sha256"] == module.sha256_file(
        embedding_manifest
    )

    ligands.write_text(ligands.read_text(encoding="utf-8") + "L9\n", encoding="utf-8")
    rejected = run_script(
        BUILD,
        [
            "--ligands",
            str(ligands),
            "--targets",
            str(targets),
            "--input-manifest",
            str(input_manifest),
            "--out-ligands",
            str(tmp_path / "tampered_ligands.csv"),
            "--out-targets",
            str(tmp_path / "tampered_targets.csv"),
            "--out-manifest",
            str(tmp_path / "tampered_embeddings.json"),
            "--fixture",
        ],
    )
    assert rejected.returncode != 0
    assert "SHA-256 mismatch" in rejected.stderr


def test_production_cli_rejects_non_preregistered_seeds(tmp_path: Path) -> None:
    embedding_manifest = build_embeddings(tmp_path)
    result = run_script(TRAIN, [
        "--train", str(write_train(tmp_path)),
        "--embedding-manifest", str(embedding_manifest),
        "--out-model", str(tmp_path / "model.pt"),
        "--out-manifest", str(tmp_path / "model.json"),
        "--out-budget", str(tmp_path / "budget.json"),
        "--seeds", "11,17,23",
        "--cpu",
    ])
    assert result.returncode != 0
    assert "production --seeds must be exactly (17, 42, 73)" in result.stderr


def test_torch_cpu_minibatched_train_and_score_is_deterministic(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    module = load_model()
    embedding_manifest, train_csv = write_torch_precomputed_inputs(tmp_path)

    def train_once(label: str) -> Path:
        model_manifest = tmp_path / f"model_{label}.json"
        result = run_script(TRAIN, [
            "--train", str(train_csv),
            "--embedding-manifest", str(embedding_manifest),
            "--out-model", str(tmp_path / f"model_{label}.pt"),
            "--out-manifest", str(model_manifest),
            "--out-budget", str(tmp_path / f"budget_{label}.json"),
            "--cpu",
            "--epochs", "1",
            "--projection-dim", "4",
            "--learning-rate", "0.001",
            "--batch-size", "2",
        ])
        assert result.returncode == 0, result.stderr
        return model_manifest

    first_manifest = train_once("first")
    second_manifest = train_once("second")
    first = json.loads(first_manifest.read_text())
    second = json.loads(second_manifest.read_text())
    assert first["training_policy"]["seeds"] == [17, 42, 73]
    assert first["model_state_sha256"] == second["model_state_sha256"]
    assert first["calibration"] == second["calibration"]
    assert first["trainable_params"] <= 10_000_000
    assert first["training_hyperparameters"]["actual_batch_size"] == 2
    assert first["training_hyperparameters"]["mixed_precision"] is False
    assert first["calibration"]["direct_binding_reversible"]["status"] == "measured_event_calibrated"
    assert first["calibration"]["functional_modulation"]["status"] == "measured_event_calibrated"
    assert first["calibration_split"]["holdout_measured_rows"] == 4
    budget = json.loads(Path(first["inputs"]["budget_manifest"]["path"]).read_text())
    assert budget["execution_mode"] == "cpu"
    assert budget["peak_vram_gib"] == 0.0
    assert budget["peak_vram_measured"] is False

    try:
        checkpoint = torch.load(first["model_artifact"]["path"], map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(first["model_artifact"]["path"], map_location="cpu")
    assert checkpoint["ligand_dim"] == 6
    assert checkpoint["target_dim"] == 4
    assert checkpoint["seeds"] == [17, 42, 73]
    assert checkpoint["seed_aggregation"] == "prediction_mean"
    assert checkpoint["model_state_sha256"] == first["model_state_sha256"]
    assert len(checkpoint["state_dicts"]) == 3
    assert all(
        not tensor.requires_grad
        for state_dict in checkpoint["state_dicts"]
        for tensor in state_dict.values()
    )
    assert not any(
        "molformer" in key.lower() or "esm" in key.lower()
        for state_dict in checkpoint["state_dicts"]
        for key in state_dict
    )
    assert len(checkpoint["embedding_feature_contract_sha256"]) == 64

    ranking_csv = tmp_path / "torch_ranking.csv"
    ranking_manifest = tmp_path / "torch_ranking.json"
    score_args = [
        "--model-manifest", str(first_manifest),
        "--ligand-id", "L1",
        "--out-ranking", str(ranking_csv),
        "--out-manifest", str(ranking_manifest),
    ]
    scored = run_script(SCORE, score_args)
    assert scored.returncode == 0, scored.stderr
    first_ranking = ranking_csv.read_text()
    scored = run_script(SCORE, score_args)
    assert scored.returncode != 0
    assert "refusing to overwrite pre-existing output" in scored.stderr
    assert ranking_csv.read_text() == first_ranking
    ranking = pd.read_csv(ranking_csv)
    assert ranking["target_id"].nunique() == 3
    assert ranking["performance_v2_ranking_score"].notna().all()
    module.validate_ranking_manifest(ranking_manifest)

    abstained_csv = tmp_path / "torch_abstained.csv"
    abstained_manifest = tmp_path / "torch_abstained.json"
    abstained_score = run_script(SCORE, [
        "--model-manifest", str(first_manifest),
        "--ligand-id", "L1",
        "--min-score", "999",
        "--out-ranking", str(abstained_csv),
        "--out-manifest", str(abstained_manifest),
    ])
    assert abstained_score.returncode == 0, abstained_score.stderr
    abstained = pd.read_csv(abstained_csv)
    assert abstained["performance_v2_abstained"].all()
    assert abstained["performance_v2_direct_binding_probability"].isna().all()
    assert abstained["performance_v2_functional_modulation_probability"].isna().all()
    module.validate_ranking_manifest(abstained_manifest)


def test_calibration_pair_identity_is_pandas_version_independent() -> None:
    module = load_model()
    frame = pd.DataFrame([
        {
            "ligand_id": ligand_id,
            "target_id": target_id,
            "evidence_state": "measured_positive" if label else "measured_negative",
            "measured_label": label,
            "ontology": ontology,
        }
        for ligand_id, target_id, ontology, label in (
            ("L1", "T1", "direct_binding_reversible", 1),
            ("L2", "T1", "direct_binding_reversible", 1),
            ("L3", "T1", "direct_binding_reversible", 0),
            ("L4", "T1", "direct_binding_reversible", 0),
            ("L5", "T2", "functional_modulation", 1),
            ("L6", "T2", "functional_modulation", 1),
            ("L7", "T2", "functional_modulation", 0),
            ("L8", "T2", "functional_modulation", 0),
        )
    ], dtype=object)

    _, _, evidence = module.calibration_holdout_split(frame, fraction=0.2)

    assert evidence["fit_pair_hashes_sha256"] == (
        "b26fab759ab381d17e651dc4ed4813b7a799b515652ae39356242c9bcdb17169"
    )
    assert evidence["holdout_pair_hashes_sha256"] == (
        "aef28453411e0683ae68fb5a1849b94f04391dcc97c4bdd091df4588e2c4df2f"
    )


def test_ranking_manifest_tamper_rejection(tmp_path: Path) -> None:
    module = load_model()
    embedding_manifest = build_embeddings(tmp_path)
    model_manifest = train_model(tmp_path, embedding_manifest)
    ranking = tmp_path / "ranking.csv"
    manifest = tmp_path / "ranking.json"
    result = run_script(SCORE, [
        "--model-manifest",
        str(model_manifest),
        "--ligand-id",
        "L1",
        "--out-ranking",
        str(ranking),
        "--out-manifest",
        str(manifest),
    ])
    assert result.returncode == 0, result.stderr
    ranking.write_text(ranking.read_text() + "\n")
    with pytest.raises(module.ContractError, match="SHA-256 mismatch"):
        module.validate_ranking_manifest(manifest)


def test_ranking_manifest_rejects_semantic_tamper_with_updated_hash(tmp_path: Path) -> None:
    module = load_model()
    embedding_manifest = build_embeddings(tmp_path)
    model_manifest = train_model(tmp_path, embedding_manifest)
    ranking = tmp_path / "ranking.csv"
    manifest = tmp_path / "ranking.json"
    result = run_script(SCORE, [
        "--model-manifest", str(model_manifest),
        "--ligand-id", "L1",
        "--out-ranking", str(ranking),
        "--out-manifest", str(manifest),
    ])
    assert result.returncode == 0, result.stderr
    frame = pd.read_csv(ranking)
    frame.loc[0, "performance_v2_ood_route"] = "dual_cold"
    frame.loc[0, "performance_v2_abstained"] = True
    frame.to_csv(ranking, index=False)
    payload = json.loads(manifest.read_text())
    payload["outputs"]["ranking_csv"] = module.artifact_record(ranking, rows=len(frame))
    module.write_json_atomic(payload, manifest)
    with pytest.raises(module.ContractError, match="OOD route does not match"):
        module.validate_ranking_manifest(manifest)


def test_model_manifest_rejects_tampered_budget_binding(tmp_path: Path) -> None:
    module = load_model()
    embedding_manifest = build_embeddings(tmp_path)
    model_manifest = train_model(tmp_path, embedding_manifest)
    payload = json.loads(model_manifest.read_text())
    budget_path = Path(payload["inputs"]["budget_manifest"]["path"])
    budget = json.loads(budget_path.read_text())
    budget["trainable_params"] = 10_000_001
    budget_path.write_text(json.dumps(budget, sort_keys=True) + "\n")
    with pytest.raises(module.ContractError, match="SHA-256 mismatch"):
        module.validate_model_manifest(model_manifest)
