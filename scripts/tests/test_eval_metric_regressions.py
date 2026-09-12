"""Small regression tests for evaluation metric semantics."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    path = ROOT / "eval" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pocket_transfer_top_one_percent_uses_active_target_universe() -> None:
    module = _load("measure_pocket_transfer")

    small = module._describe(pd.Series([1.0, 2.0, 3.0]), 100)
    large = module._describe(pd.Series([1.0, 2.0, 3.0, 20.0]), 2_000)

    assert small["top1pct_rank_cutoff"] == 1
    assert small["top1pct"] == 1
    assert large["top1pct_rank_cutoff"] == 20
    assert large["top1pct"] == 4


def test_missing_target_class_is_distinct_from_real_orphan_class() -> None:
    module = _load("disagreement_eval")

    counts = module.by_class({"known", "missing"}, {"known": "orphan"})

    assert counts == {"orphan": 1, "__missing_target_class__": 1}


def test_run_all_rejects_malformed_workflow_config_before_evaluation(
    tmp_path: Path,
) -> None:
    config = tmp_path / "malformed.yaml"
    config.write_text("evaluation: [unterminated\n")
    env = os.environ.copy()
    env.update(
        {
            "COSMAX_ROOT": str(ROOT),
            "WORKFLOW_CONFIG": str(config),
            "PYTHON_BIN": sys.executable,
            "OUT": str(tmp_path / "out"),
            "LOG": str(tmp_path / "logs"),
        }
    )

    result = subprocess.run(
        ["bash", str(ROOT / "eval" / "run_all.sh")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "invalid evaluation workflow config" in result.stderr
