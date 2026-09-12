"""Unit tests for the §13.2 retrosynthesis decision rule."""

from __future__ import annotations

import sys
import subprocess
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage7_5_score import decide, score_routes  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "stage7_5_score.py"


def test_decide_drops_zero_routes() -> None:
    assert decide(0, -1, 0.0) == "DROP"


def test_decide_downweights_long_routes() -> None:
    assert decide(2, 9, 1.0) == "DOWNWEIGHT"


def test_decide_flags_low_stock() -> None:
    assert decide(3, 4, 0.3) == "REVIEW"


def test_decide_passes_clean_route() -> None:
    assert decide(5, 4, 0.95) == "PROCEED"


def test_decide_zero_takes_precedence_over_other_signals() -> None:
    assert decide(0, 1, 1.0) == "DROP"


def test_score_routes_empty_payload() -> None:
    s = score_routes({"routes": []})
    assert s["n_routes"] == 0
    assert s["min_steps"] == -1


def test_score_routes_with_payload() -> None:
    payload = {"routes": [
        {"n_steps": 4, "score": 0.92, "stock_fraction": 1.0,
         "ra_score": 0.94, "sa_score": 2.3, "sc_score": 1.8},
        {"n_steps": 6, "score": 0.81, "stock_fraction": 0.7},
    ]}
    s = score_routes(payload)
    assert s["n_routes"] == 2
    assert s["min_steps"] == 4
    assert abs(s["best_route_score"] - 0.92) < 1e-9


def test_cli_rejects_missing_routes_json(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "analog_id\tsmiles\troutes_json\tn_routes\n"
        f"analog_0001\tCCO\t{tmp_path / 'missing.json'}\t0\n"
    )
    out_csv = tmp_path / "ranking.csv"
    out_csv.write_text("stale\n")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(manifest),
         "--out-ranking", str(out_csv)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Missing AiZynth route JSON" in result.stderr
    assert not out_csv.exists()


def test_cli_accepts_valid_zero_route_payload(tmp_path: Path) -> None:
    routes_json = tmp_path / "routes.json"
    routes_json.write_text('{"routes": []}')
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "analog_id\tsmiles\troutes_json\tn_routes\n"
        f"analog_0001\tCCO\t{routes_json}\t0\n"
    )
    out_csv = tmp_path / "ranking.csv"

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(manifest),
         "--out-ranking", str(out_csv)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    rows = out_csv.read_text().splitlines()
    assert rows[0].startswith("analog_id,smiles,n_routes")
    assert rows[1].endswith(",DROP")


def test_cli_rejects_manifest_n_routes_mismatch_without_ranking(tmp_path: Path) -> None:
    routes_json = tmp_path / "routes.json"
    routes_json.write_text(json.dumps({
        "routes": [{"n_steps": 4, "score": 0.8, "stock_fraction": 0.9}]
    }))
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "analog_id\tsmiles\troutes_json\tn_routes\n"
        f"analog_0001\tCCO\t{routes_json}\t0\n"
    )
    out_csv = tmp_path / "ranking.csv"
    out_csv.write_text("stale\n")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(manifest),
            "--out-ranking",
            str(out_csv),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "n_routes does not match route JSON" in result.stderr
    assert not out_csv.exists()


def test_cli_rejects_boolean_route_score_without_ranking(tmp_path: Path) -> None:
    routes_json = tmp_path / "routes.json"
    routes_json.write_text(json.dumps({
        "routes": [{"n_steps": 4, "score": True, "stock_fraction": 0.9}]
    }))
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "analog_id\tsmiles\troutes_json\tn_routes\n"
        f"analog_0001\tCCO\t{routes_json}\t1\n"
    )
    out_csv = tmp_path / "ranking.csv"
    out_csv.write_text("stale\n")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(manifest),
            "--out-ranking",
            str(out_csv),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "AiZynth route score for analog_0001 must be numeric" in result.stderr
    assert not out_csv.exists()


def test_cli_rejects_route_missing_score_without_ranking(tmp_path: Path) -> None:
    routes_json = tmp_path / "routes.json"
    routes_json.write_text(json.dumps({
        "routes": [{"n_steps": 4, "stock_fraction": 0.9}]
    }))
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "analog_id\tsmiles\troutes_json\tn_routes\n"
        f"analog_0001\tCCO\t{routes_json}\t1\n"
    )
    out_csv = tmp_path / "ranking.csv"
    out_csv.write_text("stale\n")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(manifest),
            "--out-ranking",
            str(out_csv),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "AiZynth route score for analog_0001 is missing" in result.stderr
    assert not out_csv.exists()


def test_cli_rejects_route_missing_n_steps_without_ranking(tmp_path: Path) -> None:
    routes_json = tmp_path / "routes.json"
    routes_json.write_text(json.dumps({
        "routes": [{"score": 0.8, "stock_fraction": 0.9}]
    }))
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "analog_id\tsmiles\troutes_json\tn_routes\n"
        f"analog_0001\tCCO\t{routes_json}\t1\n"
    )
    out_csv = tmp_path / "ranking.csv"
    out_csv.write_text("stale\n")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(manifest),
            "--out-ranking",
            str(out_csv),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "AiZynth route n_steps for analog_0001 is missing" in result.stderr
    assert not out_csv.exists()


def test_cli_rejects_route_missing_stock_fraction_without_ranking(tmp_path: Path) -> None:
    routes_json = tmp_path / "routes.json"
    routes_json.write_text(json.dumps({
        "routes": [{"n_steps": 4, "score": 0.8}]
    }))
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "analog_id\tsmiles\troutes_json\tn_routes\n"
        f"analog_0001\tCCO\t{routes_json}\t1\n"
    )
    out_csv = tmp_path / "ranking.csv"
    out_csv.write_text("stale\n")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(manifest),
            "--out-ranking",
            str(out_csv),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "AiZynth route stock_fraction for analog_0001 is missing" in result.stderr
    assert not out_csv.exists()


def test_cli_rejects_manifest_missing_required_columns(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("analog_id\tsmiles\nanalog_0001\tCCO\n")
    out_csv = tmp_path / "ranking.csv"
    out_csv.write_text("stale\n")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(manifest),
         "--out-ranking", str(out_csv)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "missing required columns" in result.stderr
    assert not out_csv.exists()


def test_cli_rejects_blank_manifest_required_fields(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "analog_id\tsmiles\troutes_json\tn_routes\n"
        f"analog_0001\t \t{tmp_path / 'routes.json'}\t0\n"
    )
    out_csv = tmp_path / "ranking.csv"
    out_csv.write_text("stale\n")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(manifest),
            "--out-ranking",
            str(out_csv),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "AiZynth manifest column 'smiles' contains blank values" in result.stderr
    assert not out_csv.exists()


def test_cli_rejects_duplicate_manifest_analog_ids(tmp_path: Path) -> None:
    route1 = tmp_path / "routes1.json"
    route2 = tmp_path / "routes2.json"
    route1.write_text('{"routes": []}')
    route2.write_text('{"routes": []}')
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "analog_id\tsmiles\troutes_json\tn_routes\n"
        f"analog_0001\tCCO\t{route1}\t0\n"
        f"analog_0001\tCCC\t{route2}\t0\n"
    )
    out_csv = tmp_path / "ranking.csv"
    out_csv.write_text("stale\n")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(manifest),
            "--out-ranking",
            str(out_csv),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "AiZynth manifest contains duplicate analog_id values: analog_0001" in result.stderr
    assert not out_csv.exists()


def test_cli_rejects_duplicate_manifest_route_json_paths(tmp_path: Path) -> None:
    routes_json = tmp_path / "routes.json"
    routes_json.write_text('{"routes": []}')
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "analog_id\tsmiles\troutes_json\tn_routes\n"
        f"analog_0001\tCCO\t{routes_json}\t0\n"
        f"analog_0002\tCCC\t{routes_json.parent}/./{routes_json.name}\t0\n"
    )
    out_csv = tmp_path / "ranking.csv"
    out_csv.write_text("stale\n")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(manifest),
            "--out-ranking",
            str(out_csv),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "AiZynth manifest contains duplicate routes_json values" in result.stderr
    assert str(routes_json.resolve()) in result.stderr
    assert not out_csv.exists()
