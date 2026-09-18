from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path



ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "stage5_6_finetune.py"


def write_smi(path: Path, smiles: list[str]) -> None:
    path.write_text("\n".join(smiles) + "\n")


def write_fake_trainer(path: Path, *, fail: bool = False) -> None:
    body = """\
import os
import pathlib
import sys

out = pathlib.Path(os.environ["TRAIN_OUT"])
if {fail!r}:
    print("trainer failed", file=sys.stderr)
    raise SystemExit(7)
(out / "agent.ckpt").write_text("agent:" + os.environ["REINVENT_SEED"])
(out / "staged.toml").write_text("run_type = 'transfer_learning'\\n")
(out / "plugins.json").write_text('{{"plugins": ["skinscout"]}}\\n')
""".format(fail=fail)
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def run(tmp_path: Path, *, allow_empty: bool = False, fail: bool = False):
    train = tmp_path / "train.smi"
    holdout = tmp_path / "holdout.smi"
    write_smi(train, ["CCO", "CCN"])
    write_smi(holdout, ["c1ccccc1"])
    prior = tmp_path / "prior.prior"
    prior.write_text("prior")
    out_dir = tmp_path / "model"
    trainer = tmp_path / "trainer.py"
    write_fake_trainer(trainer, fail=fail)
    agent = out_dir / "agent.ckpt"
    staged = out_dir / "staged.toml"
    plugins = out_dir / "plugins.json"
    manifest = tmp_path / "manifest.json"
    env = os.environ.copy()
    env["TRAIN_OUT"] = str(out_dir)
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--train-smi", str(train),
        "--holdout-smi", str(holdout),
        "--prior", str(prior),
        "--agent", str(agent),
        "--staged-config", str(staged),
        "--plugin-manifest", str(plugins),
        "--out-dir", str(out_dir),
        "--out-manifest", str(manifest),
        "--command-template", f"{sys.executable} {trainer}",
        "--seed", "17",
    ]
    if allow_empty:
        cmd.append("--allow-empty-output")
    return subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True)


def test_finetune_records_reproducible_model_contract(tmp_path: Path) -> None:
    result = run(tmp_path)
    assert result.returncode == 0, result.stderr
    payload = json.loads((tmp_path / "manifest.json").read_text())
    assert payload["execution_status"] == "ready"
    assert payload["claim_eligible"] is True
    assert payload["seed"] == 17
    assert payload["train_smiles_count"] == 2
    assert all(len(item["sha256"]) == 64 for item in payload["artifacts"])


def test_finetune_failure_is_fail_closed_and_can_be_diagnostic(tmp_path: Path) -> None:
    result = run(tmp_path, fail=True, allow_empty=True)
    assert result.returncode == 0, result.stderr
    payload = json.loads((tmp_path / "manifest.json").read_text())
    assert payload["execution_status"] == "blocked"
    assert payload["claim_eligible"] is False


def test_finetune_failure_removes_stale_manifest_without_diagnostic(tmp_path: Path) -> None:
    stale = tmp_path / "manifest.json"
    stale.write_text("stale")
    result = run(tmp_path, fail=True)
    assert result.returncode != 0
    assert not stale.exists()


def test_finetune_rejects_train_holdout_overlap(tmp_path: Path) -> None:
    train = tmp_path / "train.smi"
    holdout = tmp_path / "holdout.smi"
    write_smi(train, ["CCO"])
    write_smi(holdout, ["OCC"])
    prior = tmp_path / "prior"
    prior.write_text("prior")
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--train-smi", str(train), "--holdout-smi", str(holdout),
            "--prior", str(prior), "--agent", str(tmp_path / "agent"),
            "--staged-config", str(tmp_path / "staged"),
            "--plugin-manifest", str(tmp_path / "plugins"),
            "--out-dir", str(tmp_path / "out"),
            "--out-manifest", str(tmp_path / "manifest.json"),
            "--command-template", "true",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "overlap" in result.stderr
