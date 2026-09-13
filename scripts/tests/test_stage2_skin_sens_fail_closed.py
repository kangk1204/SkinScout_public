"""Regression tests for Stage 2 skin-sens model availability gates."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "scripts"))

from stage2_husspred import _parse_husspred_response  # noqa: E402
from stage2_pred_skin import _parse_pred_skin_response  # noqa: E402
from stage2_stoptox import _parse_stoptox_html  # noqa: E402


def write_input_sdf(path: Path) -> None:
    mol = Chem.MolFromSmiles("CCO")
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()


def run_with_broken_requests(
    tmp_path: Path,
    script: str,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    input_sdf = tmp_path / f"{script}.sdf"
    write_input_sdf(input_sdf)

    shim_dir = tmp_path / f"{script}_shim"
    shim_dir.mkdir()
    (shim_dir / "requests.py").write_text(
        "def post(*args, **kwargs):\n"
        "    raise RuntimeError('forced requests failure')\n"
    )

    env_pythonpath = os.environ.get("PYTHONPATH", "")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{shim_dir}{os.pathsep}{env_pythonpath}" if env_pythonpath else str(shim_dir)
    env["PATH"] = str(tmp_path / "empty_path")
    (tmp_path / "empty_path").mkdir(exist_ok=True)
    (tmp_path / f"{script}.json").write_text("stale\n")

    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / script),
            "--in-sdf",
            str(input_sdf),
            "--out-json",
            str(tmp_path / f"{script}.json"),
            *(extra_args or []),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def run_with_missing_requests(
    tmp_path: Path,
    script: str,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    input_sdf = tmp_path / f"{script}.sdf"
    write_input_sdf(input_sdf)

    shim_dir = tmp_path / f"{script}_missing_requests"
    shim_dir.mkdir()
    (shim_dir / "requests.py").write_text(
        "raise ImportError('forced missing requests')\n"
    )

    env_pythonpath = os.environ.get("PYTHONPATH", "")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{shim_dir}{os.pathsep}{env_pythonpath}" if env_pythonpath else str(shim_dir)
    env["PATH"] = str(tmp_path / "empty_path")
    (tmp_path / "empty_path").mkdir(exist_ok=True)
    (tmp_path / f"{script}.json").write_text("stale\n")

    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / script),
            "--in-sdf",
            str(input_sdf),
            "--out-json",
            str(tmp_path / f"{script}.json"),
            *(extra_args or []),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def run_with_mocked_requests(
    tmp_path: Path,
    script: str,
    response_payloads: list[dict],
) -> subprocess.CompletedProcess[str]:
    input_sdf = tmp_path / f"{script}.sdf"
    write_input_sdf(input_sdf)

    shim_dir = tmp_path / f"{script}_shim"
    shim_dir.mkdir()
    (shim_dir / "requests.py").write_text(
        "PAYLOADS = "
        + repr(response_payloads)
        + "\n"
        "class Response:\n"
        "    def __init__(self, payload):\n"
        "        self._payload = payload\n"
        "    def raise_for_status(self):\n"
        "        return None\n"
        "    def json(self):\n"
        "        return self._payload\n"
        "def post(*args, **kwargs):\n"
        "    if not PAYLOADS:\n"
        "        raise RuntimeError('no mocked payloads left')\n"
        "    return Response(PAYLOADS.pop(0))\n"
    )

    env_pythonpath = os.environ.get("PYTHONPATH", "")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{shim_dir}{os.pathsep}{env_pythonpath}" if env_pythonpath else str(shim_dir)
    env["PATH"] = str(tmp_path / "empty_path")
    (tmp_path / "empty_path").mkdir(exist_ok=True)
    (tmp_path / f"{script}.json").write_text("stale\n")

    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / script),
            "--in-sdf",
            str(input_sdf),
            "--out-json",
            str(tmp_path / f"{script}.json"),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_husspred_unavailable_fails_by_default(tmp_path: Path) -> None:
    res = run_with_broken_requests(tmp_path, "stage2_husspred.py")
    assert res.returncode != 0
    assert "HuSSPred is required" in res.stderr
    assert not (tmp_path / "stage2_husspred.py.json").exists()


def test_husspred_degraded_mode_is_recorded(tmp_path: Path) -> None:
    res = run_with_broken_requests(tmp_path, "stage2_husspred.py", ["--allow-unavailable"])
    assert res.returncode == 0, res.stderr
    payload = json.loads((tmp_path / "stage2_husspred.py.json").read_text())
    assert payload["status"] == "unavailable"
    assert payload["degraded"] is True
    assert payload["degraded_reason"] == "husspred_unavailable"


def test_husspred_missing_requests_can_emit_degraded_output(tmp_path: Path) -> None:
    res = run_with_missing_requests(
        tmp_path,
        "stage2_husspred.py",
        ["--allow-unavailable"],
    )
    assert res.returncode == 0, res.stderr
    payload = json.loads((tmp_path / "stage2_husspred.py.json").read_text())
    assert payload["status"] == "unavailable"
    assert payload["degraded_reason"] == "husspred_unavailable"


def test_husspred_rejects_boolean_probability_without_stale_output(
    tmp_path: Path,
) -> None:
    res = run_with_mocked_requests(
        tmp_path,
        "stage2_husspred.py",
        [{"probability_positive": True}],
    )

    assert res.returncode != 0
    assert "HuSSPred probability_positive must be numeric" in res.stderr
    assert not (tmp_path / "stage2_husspred.py.json").exists()


def test_husspred_parses_current_web_pred_data_format() -> None:
    probability, call = _parse_husspred_response(
        {
            "SMILES": "CCO",
            "pred_data": [
                [
                    "Binary Sensitization",
                    "Non-sensitizer",
                    0.86,
                    "Inside",
                    "",
                    "black",
                    "",
                ]
            ],
        }
    )

    assert call == "negative"
    assert abs(probability - 0.14) < 1e-12


def test_stoptox_unavailable_fails_by_default(tmp_path: Path) -> None:
    res = run_with_broken_requests(tmp_path, "stage2_stoptox.py")
    assert res.returncode != 0
    assert "STopTox is required" in res.stderr
    assert not (tmp_path / "stage2_stoptox.py.json").exists()


def test_stoptox_degraded_mode_is_recorded(tmp_path: Path) -> None:
    res = run_with_broken_requests(tmp_path, "stage2_stoptox.py", ["--allow-unavailable"])
    assert res.returncode == 0, res.stderr
    payload = json.loads((tmp_path / "stage2_stoptox.py.json").read_text())
    assert payload["status"] == "unavailable"
    assert payload["degraded"] is True
    assert payload["degraded_reason"] == "stoptox_unavailable"


def test_stoptox_missing_requests_can_emit_degraded_output(tmp_path: Path) -> None:
    res = run_with_missing_requests(
        tmp_path,
        "stage2_stoptox.py",
        ["--allow-unavailable"],
    )
    assert res.returncode == 0, res.stderr
    payload = json.loads((tmp_path / "stage2_stoptox.py.json").read_text())
    assert payload["status"] == "unavailable"
    assert payload["degraded_reason"] == "stoptox_unavailable"


def test_stoptox_rejects_boolean_probability_without_stale_output(
    tmp_path: Path,
) -> None:
    res = run_with_mocked_requests(
        tmp_path,
        "stage2_stoptox.py",
        [{"skin_sens": {"probability": True, "call": "negative"}}],
    )

    assert res.returncode != 0
    assert "STopTox skin_sens.probability must be numeric" in res.stderr
    assert not (tmp_path / "stage2_stoptox.py.json").exists()


def test_stoptox_parses_current_web_html_skin_sens_row() -> None:
    payload = _parse_stoptox_html(
        """
        <table>
          <tr><th>Endpoint</th><th>Prediction</th><th>Confidence</th></tr>
          <tr>
            <td><p><strong>Skin Sensitization</strong></p></td>
            <td class="Non-Toxic (-)">Non-Toxic (-)</td>
            <td>87.0%</td>
          </tr>
        </table>
        """
    )

    assert payload["skin_sens"]["call"] == "negative"
    assert abs(payload["skin_sens"]["probability"] - 0.13) < 1e-12


def test_stoptox_parses_current_web_non_sensitizer_label() -> None:
    payload = _parse_stoptox_html(
        """
        <table>
          <tr><th>Endpoint</th><th>Prediction</th><th>Confidence</th></tr>
          <tr>
            <td><p><strong>Skin Sensitization</strong></p></td>
            <td class="Non-Sensitizer (-)">Non-Sensitizer (-)</td>
            <td>91.0%</td>
          </tr>
        </table>
        """
    )

    assert payload["skin_sens"]["call"] == "negative"
    assert abs(payload["skin_sens"]["probability"] - 0.09) < 1e-12


def test_pred_skin_unavailable_fails_by_default(tmp_path: Path) -> None:
    res = run_with_broken_requests(tmp_path, "stage2_pred_skin.py")
    assert res.returncode != 0
    assert "Pred-Skin is required" in res.stderr
    assert not (tmp_path / "stage2_pred_skin.py.json").exists()


def test_pred_skin_degraded_mode_is_recorded(tmp_path: Path) -> None:
    res = run_with_broken_requests(tmp_path, "stage2_pred_skin.py", ["--allow-unavailable"])
    assert res.returncode == 0, res.stderr
    payload = json.loads((tmp_path / "stage2_pred_skin.py.json").read_text())
    assert payload["status"] == "unavailable"
    assert payload["degraded"] is True
    assert payload["degraded_reason"] == "pred_skin_unavailable"


def test_pred_skin_missing_requests_can_emit_degraded_output(tmp_path: Path) -> None:
    res = run_with_missing_requests(
        tmp_path,
        "stage2_pred_skin.py",
        ["--allow-unavailable"],
    )
    assert res.returncode == 0, res.stderr
    payload = json.loads((tmp_path / "stage2_pred_skin.py.json").read_text())
    assert payload["status"] == "unavailable"
    assert payload["degraded_reason"] == "pred_skin_unavailable"


def test_pred_skin_rejects_boolean_probability_without_stale_output(
    tmp_path: Path,
) -> None:
    res = run_with_mocked_requests(
        tmp_path,
        "stage2_pred_skin.py",
        [
            {"probability": True},
            {"probability": 0.2},
        ],
    )

    assert res.returncode != 0
    assert "pred_skin probability must be numeric" in res.stderr
    assert not (tmp_path / "stage2_pred_skin.py.json").exists()


def test_pred_skin_parses_current_task_result_format() -> None:
    probability, call, result = _parse_pred_skin_response(
        {
            "result": {
                "individual_predictions": {
                    "student": {
                        "prediction": "NC",
                        "probability": 0.04,
                    }
                }
            }
        }
    )

    assert call == "negative"
    assert probability == 0.04
    assert "individual_predictions" in result


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload))


def run_consensus(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage2_consensus.py"),
            "--admet-ai",
            str(tmp_path / "admet.json"),
            "--stoptox",
            str(tmp_path / "stoptox.json"),
            "--husspred",
            str(tmp_path / "husspred.json"),
            "--pred-skin",
            str(tmp_path / "pred_skin.json"),
            "--alerts",
            str(tmp_path / "alerts.json"),
            "--out-report",
            str(tmp_path / "report.json"),
            "--out-decision",
            str(tmp_path / "decision.txt"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def write_consensus_inputs(tmp_path: Path) -> None:
    write_json(tmp_path / "admet.json", {"smiles": "CCO", "predictions": {}})
    write_json(
        tmp_path / "husspred.json",
        {"smiles": "CCO", "skin_sens_call": "negative"},
    )
    write_json(
        tmp_path / "stoptox.json",
        {"smiles": "CCO", "skin_sens_call": "negative"},
    )
    write_json(
        tmp_path / "pred_skin.json",
        {"smiles": "CCO", "consensus_call": "negative"},
    )
    write_json(tmp_path / "alerts.json", {"smiles": "CCO"})


def test_consensus_rejects_empty_json_input_without_stale_outputs(tmp_path: Path) -> None:
    write_consensus_inputs(tmp_path)
    (tmp_path / "stoptox.json").write_text("")
    (tmp_path / "report.json").write_text("stale\n")
    (tmp_path / "decision.txt").write_text("stale\n")

    res = run_consensus(tmp_path)

    assert res.returncode != 0
    assert "Stage 2 consensus input is missing or empty: stoptox=" in res.stderr
    assert not (tmp_path / "report.json").exists()
    assert not (tmp_path / "decision.txt").exists()


def test_consensus_rejects_invalid_json_input_without_stale_outputs(tmp_path: Path) -> None:
    write_consensus_inputs(tmp_path)
    (tmp_path / "pred_skin.json").write_text("{not-json")
    (tmp_path / "report.json").write_text("stale\n")
    (tmp_path / "decision.txt").write_text("stale\n")

    res = run_consensus(tmp_path)

    assert res.returncode != 0
    assert "Stage 2 consensus input is invalid JSON: pred_skin=" in res.stderr
    assert not (tmp_path / "report.json").exists()
    assert not (tmp_path / "decision.txt").exists()


def test_consensus_rejects_non_object_json_input_without_stale_outputs(tmp_path: Path) -> None:
    write_consensus_inputs(tmp_path)
    (tmp_path / "alerts.json").write_text("[]")
    (tmp_path / "report.json").write_text("stale\n")
    (tmp_path / "decision.txt").write_text("stale\n")

    res = run_consensus(tmp_path)

    assert res.returncode != 0
    assert "Stage 2 consensus input must be a JSON object: alerts=" in res.stderr
    assert not (tmp_path / "report.json").exists()
    assert not (tmp_path / "decision.txt").exists()


def test_consensus_rejects_mismatched_input_smiles_without_stale_outputs(
    tmp_path: Path,
) -> None:
    write_consensus_inputs(tmp_path)
    write_json(
        tmp_path / "pred_skin.json",
        {"smiles": "CCN", "consensus_call": "negative"},
    )
    (tmp_path / "report.json").write_text("stale\n")
    (tmp_path / "decision.txt").write_text("stale\n")

    res = run_consensus(tmp_path)

    assert res.returncode != 0
    assert "Stage 2 consensus input smiles mismatch" in res.stderr
    assert "pred_skin=CCN" in res.stderr
    assert not (tmp_path / "report.json").exists()
    assert not (tmp_path / "decision.txt").exists()


def test_consensus_rejects_missing_skin_sens_evidence_by_default(tmp_path: Path) -> None:
    admet = tmp_path / "admet.json"
    husspred = tmp_path / "husspred.json"
    stoptox = tmp_path / "stoptox.json"
    pred_skin = tmp_path / "pred_skin.json"
    alerts = tmp_path / "alerts.json"
    write_json(admet, {"smiles": "CCO", "predictions": {}})
    write_json(husspred, {"smiles": "CCO", "status": "unavailable"})
    write_json(stoptox, {"smiles": "CCO", "skin_sens_call": "negative"})
    write_json(pred_skin, {"smiles": "CCO", "consensus_call": "negative"})
    write_json(alerts, {"smiles": "CCO"})
    (tmp_path / "report.json").write_text("stale\n")
    (tmp_path / "decision.txt").write_text("stale\n")

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage2_consensus.py"),
            "--admet-ai",
            str(admet),
            "--stoptox",
            str(stoptox),
            "--husspred",
            str(husspred),
            "--pred-skin",
            str(pred_skin),
            "--alerts",
            str(alerts),
            "--out-report",
            str(tmp_path / "report.json"),
            "--out-decision",
            str(tmp_path / "decision.txt"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "Missing Stage 2 skin-sens model evidence: husspred" in res.stderr
    assert not (tmp_path / "report.json").exists()
    assert not (tmp_path / "decision.txt").exists()


def test_consensus_rejects_invalid_skin_sens_call_without_stale_outputs(
    tmp_path: Path,
) -> None:
    write_consensus_inputs(tmp_path)
    write_json(
        tmp_path / "pred_skin.json",
        {"smiles": "CCO", "consensus_call": "maybe"},
    )
    (tmp_path / "report.json").write_text("stale\n")
    (tmp_path / "decision.txt").write_text("stale\n")

    res = run_consensus(tmp_path)

    assert res.returncode != 0
    assert (
        "Stage 2 consensus input has invalid skin-sens call: "
        "pred_skin.consensus_call='maybe'"
    ) in res.stderr
    assert not (tmp_path / "report.json").exists()
    assert not (tmp_path / "decision.txt").exists()


def test_consensus_rejects_out_of_range_skin_sens_probability_without_stale_outputs(
    tmp_path: Path,
) -> None:
    write_consensus_inputs(tmp_path)
    write_json(
        tmp_path / "husspred.json",
        {"smiles": "CCO", "skin_sens_probability": 1.2},
    )
    (tmp_path / "report.json").write_text("stale\n")
    (tmp_path / "decision.txt").write_text("stale\n")

    res = run_consensus(tmp_path)

    assert res.returncode != 0
    assert (
        "Stage 2 consensus input has out-of-range skin-sens probability: "
        "husspred.skin_sens_probability=1.2"
    ) in res.stderr
    assert not (tmp_path / "report.json").exists()
    assert not (tmp_path / "decision.txt").exists()
