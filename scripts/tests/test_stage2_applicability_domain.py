"""Regression tests for HuSSPred applicability-domain handling."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from stage2_husspred import _parse_husspred_applicability  # noqa: E402


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run_consensus(
    tmp_path: Path,
    *,
    husspred_call: str,
    husspred_ad: str | None,
    husspred_status: str = "ok",
    stoptox_call: str = "negative",
    pred_skin_call: str = "negative",
    allow_degraded: bool = False,
    admet_predictions: dict[str, float] | None = None,
) -> subprocess.CompletedProcess[str]:
    applicability = (
        {
            "status": husspred_ad,
            "raw_value": husspred_ad.title(),
            "endpoint": "Binary Sensitization",
            "source": "pred_data[0][3]",
            "requested": True,
        }
        if husspred_ad is not None
        else None
    )
    husspred: dict[str, object] = {
        "smiles": "CCO",
        "status": husspred_status,
    }
    if husspred_status == "ok":
        husspred["skin_sens_call"] = husspred_call
        husspred["skin_sens_probability"] = (
            0.8 if husspred_call == "positive" else 0.2
        )
    if applicability is not None:
        husspred["applicability_domain"] = applicability
    _write(tmp_path / "husspred.json", husspred)
    _write(tmp_path / "stoptox.json", {
        "smiles": "CCO", "status": "ok", "skin_sens_call": stoptox_call,
    })
    _write(tmp_path / "pred_skin.json", {
        "smiles": "CCO", "status": "ok", "consensus_call": pred_skin_call,
    })
    _write(
        tmp_path / "admet.json",
        {
            "smiles": "CCO",
            "status": "ok",
            "predictions": (
                {"Skin_Reaction": 0.1}
                if admet_predictions is None
                else admet_predictions
            ),
        },
    )
    _write(tmp_path / "alerts.json", {"smiles": "CCO"})
    args = [
        sys.executable,
        str(ROOT / "scripts" / "stage2_consensus.py"),
        "--admet-ai", str(tmp_path / "admet.json"),
        "--stoptox", str(tmp_path / "stoptox.json"),
        "--husspred", str(tmp_path / "husspred.json"),
        "--pred-skin", str(tmp_path / "pred_skin.json"),
        "--alerts", str(tmp_path / "alerts.json"),
        "--out-report", str(tmp_path / "report.json"),
        "--out-decision", str(tmp_path / "decision.txt"),
    ]
    if allow_degraded:
        args.append("--allow-unavailable-models")
    return subprocess.run(args, text=True, capture_output=True, check=False)


def test_real_shape_binary_row_preserves_outside_applicability() -> None:
    applicability = _parse_husspred_applicability({
        "pred_data": [[
            "Binary Sensitization", "Non-sensitizer", 0.6621,
            "Outside", "", "black", "",
        ]],
    })

    assert applicability == {
        "status": "outside",
        "raw_value": "Outside",
        "endpoint": "Binary Sensitization",
        "source": "pred_data[0][3]",
        "requested": True,
    }


def test_outside_negative_raises_applicability_gate_not_missing_gate(
    tmp_path: Path,
) -> None:
    result = _run_consensus(
        tmp_path,
        husspred_call="negative",
        husspred_ad="outside",
    )

    assert result.returncode != 0
    assert "applicability limits block a safety PASS: husspred" in result.stderr
    assert "Missing Stage 2 skin-sens model evidence" not in result.stderr
    assert not (tmp_path / "report.json").exists()


def test_outside_negative_cannot_produce_pass_in_degraded_mode(tmp_path: Path) -> None:
    result = _run_consensus(
        tmp_path,
        husspred_call="negative",
        husspred_ad="outside",
        allow_degraded=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "report.json").read_text())
    skin_sens = report["skin_sens"]
    assert skin_sens["decision"] == "FLAG_HIGH"
    assert skin_sens["husspred"] is None
    assert skin_sens["applicability_limited_models"] == ["husspred"]
    assert skin_sens["missing_models"] == []
    assert skin_sens["model_availability"]["husspred"] == "available"
    assert skin_sens["model_applicability"]["husspred"] == "outside"
    assert skin_sens["evidence_status"] == "applicability_limited"
    assert skin_sens["pass_policy"]["outside_or_unknown_applicability_negative"] == (
        "blocked_from_pass_basis"
    )
    assert skin_sens["husspred_applicability_domain"]["status"] == "outside"


def test_unavailable_husspred_is_missing_not_applicability_limited(
    tmp_path: Path,
) -> None:
    result = _run_consensus(
        tmp_path,
        husspred_call="negative",
        husspred_ad="outside",
        husspred_status="unavailable",
        allow_degraded=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "report.json").read_text())
    skin_sens = report["skin_sens"]
    assert skin_sens["decision"] == "FLAG_HIGH"
    assert skin_sens["missing_models"] == ["husspred"]
    assert skin_sens["applicability_limited_models"] == []
    assert skin_sens["model_availability"]["husspred"] == "unavailable"
    assert skin_sens["evidence_status"] == "unavailable_models"
    assert report["skin_sens"]["husspred"] is None


def test_unavailable_husspred_raises_the_missing_model_gate(tmp_path: Path) -> None:
    result = _run_consensus(
        tmp_path,
        husspred_call="negative",
        husspred_ad=None,
        husspred_status="unavailable",
    )

    assert result.returncode != 0
    assert "Missing Stage 2 skin-sens model evidence: husspred" in result.stderr
    assert not (tmp_path / "report.json").exists()


def test_inside_negative_passes_without_degraded_mode(tmp_path: Path) -> None:
    result = _run_consensus(
        tmp_path,
        husspred_call="negative",
        husspred_ad="inside",
    )

    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "report.json").read_text())
    skin_sens = report["skin_sens"]
    assert skin_sens["decision"] == "PASS"
    assert skin_sens["missing_models"] == []
    assert skin_sens["applicability_limited_models"] == []
    assert skin_sens["model_availability"]["husspred"] == "available"
    assert skin_sens["evidence_status"] == "complete"


def test_outside_positive_vote_is_retained_conservatively(tmp_path: Path) -> None:
    result = _run_consensus(
        tmp_path,
        husspred_call="positive",
        husspred_ad="outside",
        stoptox_call="positive",
        allow_degraded=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["skin_sens"]["husspred"] == "positive"
    assert report["skin_sens"]["decision"] == "HALT"


def test_missing_applicability_is_unknown_and_cannot_produce_pass(tmp_path: Path) -> None:
    result = _run_consensus(
        tmp_path,
        husspred_call="negative",
        husspred_ad=None,
        allow_degraded=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["skin_sens"]["decision"] == "FLAG_HIGH"
    assert report["skin_sens"]["husspred_applicability_domain"] is None


def test_empty_admet_predictions_are_missing_not_applicability_limited(
    tmp_path: Path,
) -> None:
    """ADMET availability is its own axis: an empty predictions dict is missing
    evidence, never an applicability limit of HuSSPred."""
    result = _run_consensus(
        tmp_path,
        husspred_call="negative",
        husspred_ad="inside",
        allow_degraded=True,
        admet_predictions={},
    )

    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "report.json").read_text())
    skin_sens = report["skin_sens"]
    assert skin_sens["missing_models"] == ["admet_ai"]
    assert skin_sens["applicability_limited_models"] == []
    assert skin_sens["model_availability"]["admet_ai"] == "unavailable"
    assert skin_sens["evidence_status"] == "unavailable_models"
    assert skin_sens["decision"] == "FLAG_HIGH"
