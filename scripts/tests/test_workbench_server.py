"""Focused tests for the local SkinScout Workbench contract."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from workbench import server
from workbench.coordinator import DurableCoordinator


ROOT = Path(__file__).resolve().parents[2]

_MINIMAL_RECEPTOR_PDB = """\
ATOM      1  N   ALA A   1      11.104   6.134  -6.504  1.00  0.00           N
ATOM      2  CA  ALA A   1      11.639   6.071  -5.147  1.00  0.00           C
ATOM      3  C   ALA A   1      13.146   6.243  -5.155  1.00  0.00           C
ATOM      4  O   ALA A   1      13.708   6.923  -6.013  1.00  0.00           O
ATOM      5  CB  ALA A   1      11.006   7.144  -4.276  1.00  0.00           C
TER       6      ALA A   1
END
"""


@pytest.fixture
def isolated_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> DurableCoordinator:
    coordinator = DurableCoordinator(tmp_path / "state" / "coordinator.sqlite3")
    monkeypatch.setattr(server, "_COORDINATOR", coordinator)
    monkeypatch.setattr(server, "_WORKER_API_TOKEN", None)
    monkeypatch.setattr(server, "_WORKER_API_TOKEN_FINGERPRINT", None)
    monkeypatch.setattr(
        server,
        "WORKER_API_TOKEN_PATH",
        tmp_path / "secrets" / "worker-api.token",
    )
    monkeypatch.setattr(server, "_JOBS", {})
    try:
        yield coordinator
    finally:
        coordinator.close()
        monkeypatch.setattr(server, "_COORDINATOR", None)


def test_safe_run_id_rejects_path_traversal() -> None:
    assert server._safe_run_id("case_01") == "case_01"
    with pytest.raises(ValueError):
        server._safe_run_id("../outside")
    with pytest.raises(ValueError):
        server._safe_run_id("case/child")


def test_target_rows_use_summary_and_ranking_artifact() -> None:
    summary = server._run_summary("ethanol_target_fast_full")
    assert summary is not None
    payload = server._target_rows("ethanol_target_fast_full", summary, query="AR", limit=5)
    assert payload["source_path"] == "results/runs/ethanol_target_fast_full/03_targets/ranked_targets_v3_with_efficacy.csv"
    assert payload["total"] >= 1
    assert payload["rows"][0]["target_id"] == "P10275"
    assert payload["rows"][0]["gene_symbol"] == "AR"


def test_target_rows_expose_preferred_hpa_cell_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "cell_context_case"
    run_dir = tmp_path / run_id
    ranking = run_dir / "03_targets" / "ranked_targets.csv"
    ranking.parent.mkdir(parents=True)
    ranking.write_text(
        "target_id,final_score,docking_rrf,skin_score,skin_tier,"
        "cell_type_preferred,source_count,sources\n"
        "P14679,0.91,0.88,0.95,high,melanocytes,3,autodock;gnina;rtmscore\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path)

    payload = server._target_rows(
        run_id,
        {"artifacts": {"target_ranking": "03_targets/ranked_targets.csv"}},
    )

    assert payload["total"] == 1
    assert payload["rows"][0]["cell_type_preferred"] == "melanocytes"


def test_legacy_artifact_adapter_rejects_unregistered_and_unsafe_paths(
    isolated_runtime: DurableCoordinator,
) -> None:
    assert server._registered_artifact_file_for_run("verified_case", "run_summary.json") is None
    assert server._registered_artifact_file_for_run(
        "verified_case",
        "../ethanol_target_fast_full/run_summary.json",
    ) is None
    assert server._registered_artifact_file_for_run("verified_case", "/etc/passwd") is None


def test_status_exposes_safe_install_boundary() -> None:
    payload = server._build_status()
    assert payload["application"] == "SkinScout Workbench"
    assert payload["csrf_token"] == server._csrf_token()
    assert payload["safe_boundary"]
    assert isinstance(payload["checks"], list)
    assert payload["system"]["platform"]
    assert set(payload["analysis_readiness"]) == {
        "safety",
        "target_fast",
        "target_comprehensive",
        "report",
        "substitute",
        "substitute_target_conditioned",
        # 대체소재 검색은 표적 예측 스테이지와 무관하게 준비될 수 있으므로 따로 센다.
        "alternatives",
    }
    assert isinstance(payload["analysis_readiness"]["target_fast"]["missing"], list)
    assert payload["disk"]["base_min_free_gb"] == server.BASE_RUNTIME_MIN_FREE_GB
    assert payload["disk"]["stage0_min_free_gb"] == server.STAGE0_MIN_FREE_GB
    assert payload["disk"]["stage0_ready"] == (
        payload["disk"]["free_gb"] >= server.STAGE0_MIN_FREE_GB
    )
    assert set(payload["setup_profiles"]) == {"demo", "full"}
    assert payload["setup_profiles"]["demo"]["default"] is True
    assert "NVIDIA 드라이버 설치는 자동으로 수행하지 않으며" in payload["safe_boundary"]


def test_demo_target_readiness_does_not_require_advanced_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "_find_command", lambda name: "/usr/bin/micromamba")
    monkeypatch.setattr(server, "_micromamba_env_exists", lambda _name: False)

    class Result:
        returncode = 0
        stdout = json.dumps(
            {
                    "tools": {
                        "gnina": {"status": "available"},
                        "autodock_gpu": {"status": "available"},
                        "autogrid": {"status": "available"},
                        "boltz": {"status": "missing"},
                    },
                "imports": {
                    "rtmscore": {"status": "missing"},
                    "psichic": {"status": "missing"},
                },
                "envs": {
                    "autodock_gpu": {"status": "declared"},
                    "meeko": {"status": "declared"},
                    "bioemu": {"status": "missing"},
                    "md": {"status": "missing"},
                    "qm": {"status": "missing"},
                },
                "gpu": {"cuda_available": True},
            }
        )

    monkeypatch.setattr(server.subprocess, "run", lambda *args, **kwargs: Result())

    payload = server._model_capabilities()

    assert payload["target_fast"] == {"ready": True, "missing": []}
    assert payload["target_comprehensive"]["ready"] is False
    assert "RTMScore" in payload["target_comprehensive"]["missing"]
    assert "PSICHIC" in payload["target_comprehensive"]["missing"]
    assert "Boltz-2" in payload["target_comprehensive"]["missing"]
    assert payload["report"]["ready"] is False
    assert "BioEmu 환경" in payload["report"]["missing"]


def test_gpu_info_blocks_analysis_when_primary_gpu_is_busy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Result:
        def __init__(self, stdout: str) -> None:
            self.returncode = 0
            self.stdout = stdout

    def fake_run(command: list[str], *args, **kwargs) -> Result:
        if any("--query-compute-apps" in item for item in command):
            return Result("")
        return Result(
            "0, GPU-test, NVIDIA GeForce RTX 3080 Ti, 12288, 11059, 852, 96\n"
        )

    monkeypatch.setattr(
        server.gpu_admission.subprocess,
        "run",
        fake_run,
    )

    payload = server._gpu_info()

    assert payload["detected"] is True
    assert payload["available_for_analysis"] is False
    assert payload["status"] == "warning"
    assert payload["free_mib"] == 852
    assert payload["utilization_percent"] == 96
    assert "다른 GPU 작업" in payload["detail"]


def test_busy_gpu_blocks_gpu_analyses_but_not_safety(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "_micromamba_env_exists", lambda _name: True)
    monkeypatch.setattr(server, "_safety_capability", lambda: {"ready": True, "missing": []})
    monkeypatch.setattr(
        server,
        "_model_capabilities",
        lambda: {
            "target_fast": {"ready": True, "missing": []},
            "target_comprehensive": {"ready": True, "missing": []},
            "report": {"ready": True, "missing": []},
        },
    )
    monkeypatch.setattr(
        server,
        "_stage0_state",
        lambda: {"status": "ready", "label": "Stage 0 데이터", "detail": "준비됨"},
    )
    monkeypatch.setattr(
        server,
        "_gpu_info",
        lambda: {
            "status": "warning",
            "label": "GPU",
            "detail": "여유 852/12,288 MiB · 사용률 96%",
            "detected": True,
            "available_for_analysis": False,
            "minimum_free_mib": server.GPU_ANALYSIS_MIN_FREE_MIB,
            "maximum_utilization_percent": server.GPU_ANALYSIS_MAX_UTILIZATION_PERCENT,
            "free_mib": 852,
            "devices": [],
            "device_details": [],
        },
    )

    payload = server._build_status()

    assert payload["analysis_readiness"]["safety"]["ready"] is True
    for key in (
        "target_fast",
        "target_comprehensive",
        "report",
        "substitute",
        "substitute_target_conditioned",
    ):
        assert payload["analysis_readiness"][key]["ready"] is False
        assert any(
            "GPU 실행 가능 상태" in item
            for item in payload["analysis_readiness"][key]["missing"]
        )
    gpu_check = next(check for check in payload["checks"] if check["id"] == "gpu-analysis")
    assert gpu_check["status"] == "warning"


def test_basic_setup_uses_beginner_conda_runtime(
    isolated_runtime: DurableCoordinator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        server,
        "get_status",
        lambda force=False: {
            "system": {"ubuntu_ready": True},
            "runtime": {"base_env": False, "p2rank": False},
        },
    )
    monkeypatch.setattr(
        server,
        "_start_process_job",
        lambda **kwargs: kwargs,
    )

    job = server._start_setup_install()

    assert job["command"][-2:] == ["--runtime", "conda"]


def test_setup_revalidates_existing_demo_runtime(
    isolated_runtime: DurableCoordinator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        server,
        "get_status",
        lambda force=False: {
            "system": {"ubuntu_ready": True},
            "runtime": {"base_env": True, "p2rank": True},
        },
    )
    monkeypatch.setattr(server, "_start_process_job", lambda **kwargs: kwargs)

    job = server._start_setup_install("demo")

    assert job["command"][-2:] == ["--runtime", "conda"]


def test_setup_full_profile_uses_target_model_flag(
    isolated_runtime: DurableCoordinator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        server,
        "get_status",
        lambda force=False: {
            "system": {"ubuntu_ready": True},
            "runtime": {"base_env": False, "p2rank": False},
        },
    )
    monkeypatch.setattr(
        server,
        "_start_process_job",
        lambda **kwargs: kwargs,
    )

    job = server._start_setup_install("full")

    assert job["command"][-4:] == ["--runtime", "conda", "--profile", "full"]


def test_setup_full_profile_upgrades_existing_demo_runtime(
    isolated_runtime: DurableCoordinator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        server,
        "get_status",
        lambda force=False: {
            "system": {"ubuntu_ready": True},
            "runtime": {"base_env": True, "p2rank": True},
        },
    )
    monkeypatch.setattr(server, "_micromamba_env_exists", lambda _name: False)
    monkeypatch.setattr(server, "_start_process_job", lambda **kwargs: kwargs)

    job = server._start_setup_install("full")

    assert job["command"][-4:] == ["--runtime", "conda", "--profile", "full"]


def test_setup_rejects_private_profile_name(
    isolated_runtime: DurableCoordinator,
) -> None:
    with pytest.raises(ValueError, match="demo 또는 full"):
        server._start_setup_install("operator")


def test_stage0_start_blocks_when_disk_is_below_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        server,
        "get_status",
        lambda force=False: {
            "stage0": {"status": "missing"},
            "disk": {"free_gb": 199.9, "stage0_ready": False},
        },
    )

    with pytest.raises(RuntimeError, match="최소 200 GB"):
        server._start_stage0(16)


def test_subprocess_environment_uses_environment_specific_mamba_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_root = tmp_path / "base-root"
    boltz_root = tmp_path / "boltz-root"
    (base_root / "envs" / "cosmax-base").mkdir(parents=True)
    (boltz_root / "envs" / "cosmax-boltz2").mkdir(parents=True)
    monkeypatch.delenv("MAMBA_ROOT_PREFIX", raising=False)
    monkeypatch.setattr(server, "_mamba_root_candidates", lambda: [base_root, boltz_root])

    assert server._subprocess_env(mamba_environment="cosmax-base")["MAMBA_ROOT_PREFIX"] == str(base_root)
    assert server._subprocess_env(mamba_environment="cosmax-boltz2")["MAMBA_ROOT_PREFIX"] == str(boltz_root)


def test_failed_safety_preflight_blocks_every_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "_micromamba_env_exists", lambda _name: True)
    monkeypatch.setattr(
        server,
        "_safety_capability",
        lambda: {"ready": False, "missing": ["env:admet-ai"]},
    )
    monkeypatch.setattr(
        server,
        "_model_capabilities",
        lambda: {
            "target_fast": {"ready": True, "missing": []},
            "target_comprehensive": {"ready": True, "missing": []},
            "report": {"ready": True, "missing": []},
        },
    )

    payload = server._build_status()
    assert payload["analysis_readiness"]["safety"]["ready"] is False
    assert payload["analysis_readiness"]["target_fast"]["ready"] is False
    assert "안전성 사전검사(env:admet-ai)" in payload["analysis_readiness"]["target_fast"]["missing"]


def test_activity_retrieval_gate_blocks_target_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "_micromamba_env_exists", lambda _name: True)
    monkeypatch.setattr(server, "_safety_capability", lambda: {"ready": True, "missing": []})
    monkeypatch.setattr(
        server,
        "_activity_retrieval_capability",
        lambda: {"ready": False, "missing": ["활성 검색 운영 gate(stale index)"]},
    )
    monkeypatch.setattr(
        server,
        "_model_capabilities",
        lambda: {
            "target_fast": {"ready": True, "missing": []},
            "target_comprehensive": {"ready": True, "missing": []},
            "report": {"ready": True, "missing": []},
        },
    )
    monkeypatch.setattr(
        server,
        "_stage0_state",
        lambda: {"status": "ready", "label": "Stage 0 데이터", "detail": "준비됨"},
    )
    monkeypatch.setattr(
        server,
        "_gpu_info",
        lambda: {
            "available_for_analysis": True,
            "detected": True,
            "status": "ready",
            "detail": "ready",
        },
    )

    payload = server._build_status()

    assert payload["analysis_readiness"]["safety"]["ready"] is True
    for key in ("target_fast", "target_comprehensive", "report", "substitute"):
        assert payload["analysis_readiness"][key]["ready"] is False
        assert "활성 검색 운영 gate(stale index)" in payload["analysis_readiness"][key]["missing"]


def test_cancel_requested_state_is_preserved_until_process_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        def __init__(self, returncode: int | None) -> None:
            self.returncode = returncode

        def poll(self) -> int | None:
            return self.returncode

    running = server.Job(
        job_id="cancel_running",
        kind="run",
        run_id="cancel_case",
        status="cancel_requested",
        process=FakeProcess(None),  # type: ignore[arg-type]
    )
    monkeypatch.setattr(server, "_JOBS", {running.job_id: running})
    server._refresh_jobs()
    assert running.status == "cancel_requested"
    assert server._active_job_for_run("cancel_case") is running

    running.process = FakeProcess(-15)  # type: ignore[assignment]
    server._refresh_jobs()
    assert running.status == "cancelled"
    assert running.returncode == -15


def test_http_server_serves_ui_and_run_api(
    isolated_runtime: DurableCoordinator,
) -> None:
    httpd = server.create_server("127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        with urlopen(base + "/", timeout=5) as response:
            html = response.read().decode("utf-8")
            assert response.headers["X-Frame-Options"] == "DENY"
            assert "'unsafe-eval'" not in response.headers["Content-Security-Policy"]
        assert "SkinScout Workbench" in html
        with urlopen(base + "/api/runs", timeout=5) as response:
            runs = json.loads(response.read().decode("utf-8"))
        assert isinstance(runs["runs"], list)
        with urlopen(base + "/api/v2/runs", timeout=5) as response:
            runs_v2 = json.loads(response.read().decode("utf-8"))
        assert runs_v2["schema_version"] == 2
        with urlopen(base + "/api/identity", timeout=5) as response:
            identity = json.loads(response.read().decode("utf-8"))
        assert identity == {"application": "SkinScout Workbench"}
        # A file that exists in the run directory downloads even though nothing
        # promoted it: no production code calls the coordinator's promote API,
        # so requiring registration meant no artifact was ever downloadable.
        with urlopen(
            base + "/api/runs/ethanol_target_fast_full/artifact?path=run_summary.json",
            timeout=5,
        ) as response:
            assert response.status == 200
            assert json.loads(response.read().decode("utf-8"))

        with pytest.raises(HTTPError) as escaped_artifact_error:
            urlopen(
                base
                + "/api/runs/ethanol_target_fast_full/artifact?path=../../../etc/passwd",
                timeout=5,
            )
        assert escaped_artifact_error.value.code == 404

        cross_origin = Request(
            base + "/api/runs",
            data=b"{}",
            headers={
                "Content-Type": "application/json",
                "Origin": "https://example.invalid",
                "X-SkinScout-Token": server._csrf_token(),
            },
            method="POST",
        )
        with pytest.raises(HTTPError) as cross_origin_error:
            urlopen(cross_origin, timeout=5)
        assert cross_origin_error.value.code == 403

        rebound_host = Request(
            base + "/api/runs",
            data=b"{}",
            headers={
                "Content-Type": "application/json",
                "Host": "attacker.invalid",
                "Origin": "http://attacker.invalid",
                "X-SkinScout-Token": server._csrf_token(),
            },
            method="POST",
        )
        with pytest.raises(HTTPError) as rebound_host_error:
            urlopen(rebound_host, timeout=5)
        assert rebound_host_error.value.code == 403

        missing_token = Request(
            base + "/api/runs",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as missing_token_error:
            urlopen(missing_token, timeout=5)
        assert missing_token_error.value.code == 403

        wrong_type = Request(
            base + "/api/runs",
            data=b"{}",
            headers={
                "Content-Type": "text/plain",
                "X-SkinScout-Token": server._csrf_token(),
            },
            method="POST",
        )
        with pytest.raises(HTTPError) as wrong_type_error:
            urlopen(wrong_type, timeout=5)
        assert wrong_type_error.value.code == 415
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_worker_api_token_permissions_and_authenticated_lifecycle(
    isolated_runtime: DurableCoordinator,
) -> None:
    coordinator = isolated_runtime
    coordinator.create_job(
        "run",
        {"run_id": "worker_case", "command": [], "log_path": None},
        job_id="worker-job",
    )
    httpd = server.create_server("127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        token_path = server.WORKER_API_TOKEN_PATH
        assert stat.S_IMODE(token_path.stat().st_mode) == 0o600
        token = token_path.read_text().strip()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"

        before = coordinator.list_events("worker-job")
        wrong = Request(
            base + "/api/v2/worker/claim",
            data=json.dumps({"worker_id": "container-1"}).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer wrong",
            },
            method="POST",
        )
        with pytest.raises(HTTPError) as wrong_error:
            urlopen(wrong, timeout=5)
        assert wrong_error.value.code == 401
        assert coordinator.list_events("worker-job") == before

        claim = Request(
            base + "/api/v2/worker/claim",
            data=json.dumps({"worker_id": "container-1"}).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
            method="POST",
        )
        with urlopen(claim, timeout=5) as response:
            claimed = json.loads(response.read().decode())
        attempt = claimed["attempt"]
        attempt_token = claimed["attempt_token"]
        assert "attempt_token" not in json.dumps(attempt)

        heartbeat = Request(
            base
            + f"/api/v2/worker/attempts/{attempt['attempt_id']}/heartbeat",
            data=json.dumps({"attempt_token": attempt_token}).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
            method="POST",
        )
        with urlopen(heartbeat, timeout=5) as response:
            heartbeat_payload = json.loads(response.read().decode())
        assert heartbeat_payload["result"]["status"] == "running"

        complete = Request(
            base + f"/api/v2/worker/attempts/{attempt['attempt_id']}/complete",
            data=json.dumps({
                "attempt_token": attempt_token,
                "status": "completed",
                "result": {"artifact": "fast"},
            }).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
            method="POST",
        )
        with urlopen(complete, timeout=5) as response:
            completed = json.loads(response.read().decode())
        assert completed["result"]["status"] == "completed"

        with urlopen(base + "/api/v2/jobs/worker-job/events", timeout=5) as response:
            events = json.loads(response.read().decode())
        assert events["schema_version"] == 2
        assert [item["seq"] for item in events["events"]] == [1, 2, 3, 4]
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_worker_promotes_registry_only_artifact_with_head_range_and_etag(
    isolated_runtime: DurableCoordinator,
) -> None:
    coordinator = isolated_runtime
    coordinator.create_job(
        "run",
        {"run_id": "artifact_case", "command": [], "log_path": None},
        job_id="artifact-worker-job",
    )
    httpd = server.create_server("127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        token = server.WORKER_API_TOKEN_PATH.read_text().strip()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        auth_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }
        claim = Request(
            base + "/api/v2/worker/claim",
            data=json.dumps({"worker_id": "container-artifact"}).encode(),
            headers=auth_headers,
            method="POST",
        )
        with urlopen(claim, timeout=5) as response:
            claimed = json.loads(response.read().decode())
        attempt = claimed["attempt"]
        attempt_token = claimed["attempt_token"]
        workspace = Path(claimed["workspace"])
        content = b"0123456789abcdef"
        workspace.joinpath("index.html").write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        manifest = {
            "schema_version": "skinscout.attempt_artifacts.v1",
            "files": [{"path": "index.html", "bytes": len(content), "sha256": digest}],
        }
        promote = Request(
            base + f"/api/v2/worker/attempts/{attempt['attempt_id']}/promote",
            data=json.dumps({
                "attempt_token": attempt_token,
                "namespace": "runs/artifact_case/reports/fast",
                "manifest": manifest,
            }).encode(),
            headers=auth_headers,
            method="POST",
        )
        with urlopen(promote, timeout=5) as response:
            promoted = json.loads(response.read().decode())["result"]
        artifact_id = promoted["artifact_id"]
        artifact_url = base + f"/api/v2/artifacts/{artifact_id}/files/index.html"

        with urlopen(base + f"/api/v2/artifacts/{artifact_id}", timeout=5) as response:
            metadata = json.loads(response.read().decode())["artifact"]
        assert metadata["files"][0]["sha256"] == digest
        assert metadata["files"][0]["download_url"].endswith("/files/index.html")

        with urlopen(artifact_url, timeout=5) as response:
            assert response.read() == content
            assert response.headers["ETag"] == f'"{digest}"'
            assert response.headers["Accept-Ranges"] == "bytes"
            assert response.headers["X-Content-Type-Options"] == "nosniff"
            assert "'unsafe-eval'" in response.headers["Content-Security-Policy"]

        legacy_adapter_url = (
            base + "/api/runs/artifact_case/artifact?path=index.html"
        )
        with urlopen(legacy_adapter_url, timeout=5) as response:
            assert response.read() == content
            assert response.headers["ETag"] == f'"{digest}"'

        head = Request(artifact_url, method="HEAD")
        with urlopen(head, timeout=5) as response:
            assert response.status == 200
            assert response.headers["Content-Length"] == str(len(content))
            assert response.read() == b""

        ranged = Request(artifact_url, headers={"Range": "bytes=3-7"})
        with urlopen(ranged, timeout=5) as response:
            assert response.status == 206
            assert response.headers["Content-Range"] == f"bytes 3-7/{len(content)}"
            assert response.read() == content[3:8]

        coordinator.complete_attempt(
            attempt["attempt_id"],
            attempt_token,
            result={"artifact_id": artifact_id},
        )
        with urlopen(base + "/api/v2/runs/artifact_case/artifacts", timeout=5) as response:
            listing = json.loads(response.read().decode())
        assert listing["artifacts"][0]["artifact_id"] == artifact_id

        traversal = base + f"/api/v2/artifacts/{artifact_id}/files/%2e%2e/index.html"
        with pytest.raises(HTTPError) as traversal_error:
            urlopen(traversal, timeout=5)
        assert traversal_error.value.code == 404
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_worker_claim_cannot_select_another_gpu_resource(
    isolated_runtime: DurableCoordinator,
) -> None:
    coordinator = isolated_runtime
    coordinator.create_job("run", {"run_id": "gpu_one"}, job_id="gpu-one")
    coordinator.create_job("run", {"run_id": "gpu_two"}, job_id="gpu-two")
    httpd = server.create_server("127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        token = server.WORKER_API_TOKEN_PATH.read_text().strip()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }
        first = Request(
            base + "/api/v2/worker/claim",
            data=json.dumps({"worker_id": "worker-one"}).encode(),
            headers=headers,
            method="POST",
        )
        with urlopen(first, timeout=5) as response:
            assert json.loads(response.read())["attempt"]["lease_resource"] == "gpu:0"

        bypass = Request(
            base + "/api/v2/worker/claim",
            data=json.dumps({"worker_id": "worker-two", "resource": "gpu:1"}).encode(),
            headers=headers,
            method="POST",
        )
        with pytest.raises(HTTPError) as bypass_error:
            urlopen(bypass, timeout=5)
        assert bypass_error.value.code == 400
        assert coordinator.get_job("gpu-two")["status"] == "queued"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_worker_secret_rotation_reloads_token_and_invalidates_leases(
    isolated_runtime: DurableCoordinator,
) -> None:
    coordinator = isolated_runtime
    old_token = server._worker_api_token()
    coordinator.create_job("run", {"run_id": "rotation"}, job_id="rotation")
    attempt = coordinator.claim_next(worker_id="worker-one")
    assert attempt is not None

    replacement = server.WORKER_API_TOKEN_PATH.with_suffix(".replacement")
    new_token = "n" * 48
    replacement.write_text(new_token + "\n", encoding="ascii")
    replacement.chmod(0o600)
    replacement.replace(server.WORKER_API_TOKEN_PATH)

    assert server._worker_authorized(f"Bearer {old_token}") is False
    assert server._worker_authorized(f"Bearer {new_token}") is True
    assert coordinator.get_job("rotation")["status"] == "queued"
    assert [event["type"] for event in coordinator.list_events("rotation")][-2:] == [
        "attempt.invalidated",
        "job.requeued",
    ]


def test_server_rejects_an_unauthenticated_non_loopback_binding(
    isolated_runtime: DurableCoordinator,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A remote bind is allowed now, but never without a login.

    Refusing it outright left an SSH tunnel as the only way to reach a lab GPU
    box; allowing it without authentication would have handed every read
    endpoint - results, rankings, artifacts - to anyone on the network.
    """
    monkeypatch.setattr(server, "ACCESS_TOKEN_PATH", tmp_path / "access.token")
    try:
        with pytest.raises(ValueError, match="인증이 필요합니다"):
            server.create_server("0.0.0.0", 0, require_auth=False)

        httpd = server.create_server("0.0.0.0", 0)
        try:
            assert server.REQUIRE_AUTH is True
        finally:
            httpd.server_close()
    finally:
        server.REQUIRE_AUTH = False


def test_process_job_lifecycle_is_persisted(
    isolated_runtime: DurableCoordinator,
    tmp_path: Path,
) -> None:
    coordinator = isolated_runtime
    job = server._start_process_job(
        kind="run",
        run_id="durable_process",
        command=[sys.executable, "-c", "print('ok')"],
        log_path=tmp_path / "durable.log",
    )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if coordinator.get_job(job.job_id)["status"] in {
            "completed",
            "failed",
            "cancelled",
        }:
            break
        time.sleep(0.02)

    assert coordinator.get_job(job.job_id)["status"] == "completed"
    assert [event["type"] for event in coordinator.list_events(job.job_id)] == [
        "job.created",
        "attempt.claimed",
        "attempt.process_started",
        "job.completed",
    ]
    assert "ok" in (tmp_path / "durable.log").read_text()


@pytest.mark.skipif(server.os.name != "posix", reason="process-group recovery is POSIX-only")
def test_restart_stops_recorded_child_before_requeue(tmp_path: Path) -> None:
    db = tmp_path / "restart.sqlite3"
    first = DurableCoordinator(db)
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    try:
        first.create_job(
            "run",
            {"run_id": "restart-case", "managed_by": "workbench-local-process"},
            job_id="restart-case",
        )
        attempt = first.claim_job("restart-case", worker_id="workbench:test")
        first.record_process_identity(
            attempt.attempt_id,
            attempt.token,
            server._process_identity(child),
        )
        first.close()

        restarted = DurableCoordinator(db)
        try:
            server._reconcile_interrupted_processes(restarted)
            child.wait(timeout=3)
            assert restarted.get_job("restart-case")["status"] == "queued"
            assert "attempt.interrupted" in [
                event["type"] for event in restarted.list_events("restart-case")
            ]
        finally:
            restarted.close()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=3)


def test_process_job_rechecks_gpu_after_acquiring_lease(
    isolated_runtime: DurableCoordinator,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = isolated_runtime
    monkeypatch.setattr(
        server,
        "_gpu_info",
        lambda: {
            "available_for_analysis": False,
            "detail": "다른 GPU 계산 1개가 실행 중입니다.",
        },
    )

    with pytest.raises(RuntimeError, match="다른 GPU 계산 1개"):
        server._start_process_job(
            kind="run",
            run_id="gpu_recheck",
            command=[sys.executable, "-c", "print('must not run')"],
            log_path=tmp_path / "gpu_recheck.log",
            require_gpu_admission=True,
        )

    record = next(
        item
        for item in coordinator.list_jobs(limit=10)
        if item["payload"]["run_id"] == "gpu_recheck"
    )
    assert record["status"] == "failed"
    assert [event["type"] for event in coordinator.list_events(record["job_id"])] == [
        "job.created",
        "attempt.claimed",
        "job.failed",
    ]


def test_workbench_run_uses_canonical_profile_resolver(
    isolated_runtime: DurableCoordinator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    status_forces: list[bool] = []

    def ready_status(force: bool = False) -> dict[str, object]:
        status_forces.append(force)
        return {
            "analysis_readiness": {
                "target_fast": {"ready": True, "missing": []},
            }
        }

    monkeypatch.setattr(
        server,
        "get_status",
        ready_status,
    )
    monkeypatch.setattr(server, "_active_job_for_run", lambda _run_id: None)

    def fake_start(**kwargs: object) -> server.Job:
        captured.update(kwargs)
        return server.Job(
            job_id="profile-job",
            kind="run",
            run_id=str(kwargs["run_id"]),
        )

    monkeypatch.setattr(server, "_start_process_job", fake_start)
    job = server._start_run({
        "input_type": "smiles",
        "smiles": "CCO",
        "run_id": "profile_case",
        "preset": "target-id-sota",
        "mode": "fast",
        "context_profile": "barrier",
    })

    assert job.job_id == "profile-job"
    command = captured["command"]
    assert isinstance(command, list)
    assert command[command.index("--preset") + 1] == "target-id-sota"
    assert "--online-safety-readiness" in command
    payload_extra = captured["payload_extra"]
    assert isinstance(payload_extra, dict)
    assert payload_extra["run_profile"]["analysis_profile"] == "target_fast"
    assert payload_extra["run_profile"]["sota_claim"] is True
    assert status_forces == [True]


def test_workbench_substitute_run_uses_verified_target_wrapper(
    isolated_runtime: DurableCoordinator,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(
        server,
        "get_status",
        lambda force=False: {
            "analysis_readiness": {
                "substitute": {"ready": True, "missing": []},
                "substitute_target_conditioned": {"ready": True, "missing": []},
            }
        },
    )
    monkeypatch.setattr(server, "_active_job_for_run", lambda _run_id: None)

    def fake_start(**kwargs: object) -> server.Job:
        captured.update(kwargs)
        return server.Job(
            job_id="substitute-job",
            kind="run",
            run_id=str(kwargs["run_id"]),
        )

    monkeypatch.setattr(server, "_start_process_job", fake_start)
    job = server._start_run({
        "input_type": "smiles",
        "smiles": "CCO",
        "run_id": "substitute_case",
        "preset": "substitute",
        "mode": "fast",
        "evidence_mode": "evidence",
        "target_id": "p14679",
        "max_candidates": 25,
    })

    assert job.job_id == "substitute-job"
    command = captured["command"]
    assert isinstance(command, list)
    assert any(Path(item).name == "run_substitute_discovery.py" for item in command)
    assert command[command.index("--target-id") + 1] == "P14679"
    assert command[command.index("--max-candidates") + 1] == "25"
    assert "--build-interaction-anchors" in command
    assert "--preset" not in command
    payload_extra = captured["payload_extra"]
    assert isinstance(payload_extra, dict)
    assert payload_extra["analysis_kind"] == "substitute"
    assert payload_extra["requested_preset"] == "substitute"
    assert payload_extra["run_profile"]["analysis_profile"] == "target_fast"
    assert payload_extra["target_conditioned"] is True


def test_workbench_substitute_can_explicitly_use_2d_only_fallback(
    isolated_runtime: DurableCoordinator,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(
        server,
        "get_status",
        lambda force=False: {
            "analysis_readiness": {
                "substitute": {"ready": True, "missing": []},
            }
        },
    )
    monkeypatch.setattr(server, "_active_job_for_run", lambda _run_id: None)

    def fake_start(**kwargs: object) -> server.Job:
        captured.update(kwargs)
        return server.Job(job_id="substitute-2d", kind="run", run_id=str(kwargs["run_id"]))

    monkeypatch.setattr(server, "_start_process_job", fake_start)

    server._start_run(
        {
            "input_type": "smiles",
            "smiles": "CCO",
            "run_id": "substitute_2d_case",
            "preset": "substitute",
            "mode": "fast",
            "evidence_mode": "evidence",
            "target_conditioned": False,
        }
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert "--build-interaction-anchors" not in command
    payload_extra = captured["payload_extra"]
    assert isinstance(payload_extra, dict)
    assert payload_extra["target_conditioned"] is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("mode", "comprehensive", "빠른 분석"),
        ("evidence_mode", "discovery", "Evidence 모드"),
        ("target_id", "not a target", "UniProt accession"),
        ("max_candidates", 0, "1에서 200"),
        ("max_candidates", True, "정수"),
        ("target_conditioned", "yes", "참/거짓"),
    ),
)
def test_workbench_substitute_run_rejects_unsupported_options(
    field: str,
    value: object,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "_active_job_for_run", lambda _run_id: None)
    monkeypatch.setattr(
        server,
        "get_status",
        lambda force=False: {
            "analysis_readiness": {
                "substitute": {"ready": True, "missing": []},
                "substitute_target_conditioned": {"ready": True, "missing": []},
            }
        },
    )
    payload: dict[str, object] = {
        "input_type": "smiles",
        "smiles": "CCO",
        "run_id": "invalid_substitute_case",
        "preset": "substitute",
        "mode": "fast",
        "evidence_mode": "evidence",
        "max_candidates": 25,
    }
    payload[field] = value

    with pytest.raises((ValueError, RuntimeError), match=message):
        server._start_run(payload)


def _valid_substitute_report(run_id: str) -> dict[str, object]:
    return {
        "schema_version": server.SUBSTITUTE_SCHEMA,
        "status": "completed",
        "run_id": run_id,
        "claimable": False,
        "hypothesis_only": True,
        "wet_lab_required": True,
        "parent": {"smiles": "CCO"},
        "target": {"target_id": "P14679"},
        "thresholds": {
            "max_direct_pactivity_loss_for_retention": 1.0,
            "min_pharmacophore_preservation_score": 0.55,
            "min_feature_family_recall": 0.6,
        },
        "candidate_pool": {"passing_scored_molecules": 3},
        "selection_strategy": {
            "mode": "balanced_tracks",
            "material_track_fraction": 0.4,
            "pharmacophore_track_fraction": 0.2,
            "reserved_material_slots": 1,
            "reserved_pharmacophore_slots": 1,
            "eligible_counts": {
                "all": 3,
                "target_activity": 2,
                "cosmetic_material": 1,
                "strict_pharmacophore": 1,
            },
            "selected_counts": {
                "all": 1,
                "target_activity": 1,
                "cosmetic_material": 1,
                "strict_pharmacophore": 1,
                "both": 1,
            },
        },
        "summary": {
            "direct_activity_candidates": 1,
            "direct_retained_candidates": 1,
            "cosing_candidates": 1,
            "proxy_only_candidates": 0,
            "strict_pharmacophore_candidates": 1,
            "target_supported_feature_analogue_candidates": 1,
            "target_supported_nonpharmacophore_candidates": 0,
        },
        "candidates": [
            {
                "rank": 1,
                "candidate_id": "SUB0001",
                "name": "candidate",
                "smiles": "CCN",
                "candidate_sources": ["cosing", "ChEMBL"],
                "cosing_reference": True,
                "selection_tracks": ["target_activity", "cosmetic_material"],
                "global_priority_rank": 1,
                "evidence_tier": "direct_retained",
                "admission_bases": [
                    "strict_2d_pharmacophore",
                    "same_target_activity_feature_family",
                ],
                "pharmacophore_gate_passed": True,
                "binding_retained": True,
                "binding_pactivity_delta": -0.3,
                "binding_comparison_basis": "same_activity_type_and_source_conservative_delta",
                "binding_comparison_strata": [
                    {
                        "activity_type": "KI",
                        "source": "ChEMBL",
                        "candidate_median_pactivity": 7.0,
                        "parent_median_pactivity": 7.3,
                        "pactivity_delta": -0.3,
                        "candidate_evidence_count": 2,
                        "parent_evidence_count": 1,
                    }
                ],
                "activity_evidence_count": 2,
                "pharmacophore_preservation_score": 0.8,
                "feature_family_recall": 0.8,
                "feature_family_precision": 0.75,
                "feature_family_f1": 0.77,
                "binding_support_score": 0.7,
                "safety_triage_score": 0.9,
                "routeability_proxy": 0.8,
                "priority_score": 0.78,
                "claimable": False,
                "hypothesis_only": True,
                "wet_lab_required": True,
            }
        ],
    }


def _write_substitute_bundle(
    runs_root: Path,
    run_id: str,
    payload: dict[str, object],
) -> Path:
    run_dir = runs_root / run_id
    substitute_dir = run_dir / server.SUBSTITUTE_RELATIVE_DIR
    substitute_dir.mkdir(parents=True, exist_ok=True)
    parent_summary = {
        "schema_version": "skinscout.run_summary.v1",
        "run_id": run_id,
        "preset": "target-id",
        "mode": "fast",
        "compound": {"canonical_smiles": "CCO"},
    }
    parent_verification = {"status": "ok", "preset": "target-id"}
    (run_dir / "run_summary.json").write_text(json.dumps(parent_summary))
    (run_dir / "run_verification.json").write_text(json.dumps(parent_verification))
    report_path = substitute_dir / "substitute_report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    for filename, content in (
        ("substitute_candidates.csv", "candidate_id,smiles\nSUB0001,CCN\n"),
        ("substitute_candidates_3d.sdf", "SkinScout fixture\n$$$$\n"),
        ("substitute_report.html", "<!doctype html><title>fixture</title>"),
        ("substitute_report.md", "# fixture\n"),
    ):
        (substitute_dir / filename).write_text(content, encoding="utf-8")

    def fingerprint(path: Path) -> dict[str, object]:
        return {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    manifest = {
        "schema_version": server.SUBSTITUTE_RUN_SCHEMA,
        "run_id": run_id,
        "claimable": False,
        "hypothesis_only": True,
        "wet_lab_required": True,
        "parent_run": {
            "preset": "target-id",
            "canonical_smiles": "CCO",
            "summary": fingerprint(run_dir / "run_summary.json"),
            "verification": fingerprint(run_dir / "run_verification.json"),
        },
        "target": {"target_id": payload["target"]["target_id"]},  # type: ignore[index]
        "outputs": {
            filename: fingerprint(substitute_dir / filename)
            for filename in server.SUBSTITUTE_OUTPUT_FILES
        },
        "candidate_count": len(payload["candidates"]),  # type: ignore[arg-type]
        "candidate_summary": payload["summary"],
        "selection_strategy": payload["selection_strategy"],
    }
    anchor_conditioned = payload["target"].get(  # type: ignore[union-attr]
        "interaction_anchor_conditioned",
        False,
    )
    if anchor_conditioned:
        anchor_path = run_dir / "05_pharmacophore" / "interaction_anchor_map.json"
        anchor_path.parent.mkdir(parents=True, exist_ok=True)
        anchor_path.write_text("{}\n", encoding="utf-8")
        manifest["interaction_anchor_map"] = fingerprint(anchor_path)
    else:
        manifest["interaction_anchor_map"] = None
    (substitute_dir / "substitute_run_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return report_path


def test_substitute_report_is_exposed_only_after_contract_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "validated_substitutes"
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path)
    payload = _valid_substitute_report(run_id)
    _write_substitute_bundle(tmp_path, run_id, payload)

    report = server._substitute_report(run_id)
    assert report is not None
    assert report["status"] == "completed"
    assert len(report["candidates"]) == 1

    payload["candidates"][0]["claimable"] = True  # type: ignore[index]
    _write_substitute_bundle(tmp_path, run_id, payload)
    blocked = server._substitute_report(run_id)
    assert blocked is not None
    assert blocked["status"] == "blocked"
    assert blocked["candidates"] == []
    assert "claim boundary" in blocked["error"]


def test_substitute_report_validates_target_conditioned_anchor_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "target_conditioned_substitutes"
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path)
    payload = _valid_substitute_report(run_id)
    payload["target"]["interaction_anchor_conditioned"] = True  # type: ignore[index]
    payload["summary"]["target_conditioned_anchor_candidates"] = 1  # type: ignore[index]
    candidate = payload["candidates"][0]  # type: ignore[index]
    candidate.update(
        {
            "target_conditioned_anchor_score": 0.75,
            "target_conditioned_anchor_count": 4,
            "target_conditioned_preserved_anchor_count": 3,
            "target_conditioned_anchor_basis": (
                "pose_supported_parent_anchor_conservative_mcs_feature_preservation"
            ),
            "target_conditioned_mapping_count": 2,
            "target_conditioned_mapping_ambiguous": True,
            "target_conditioned_mapping_truncated": False,
            "analog_pose_verified": False,
        }
    )
    _write_substitute_bundle(tmp_path, run_id, payload)

    report = server._substitute_report(run_id)

    assert report is not None
    assert report["status"] == "completed"
    candidate["target_conditioned_anchor_score"] = 1.5
    _write_substitute_bundle(tmp_path, run_id, payload)
    blocked = server._substitute_report(run_id)
    assert blocked is not None
    assert blocked["status"] == "blocked"
    assert "target_conditioned_anchor" in blocked["error"]


def test_substitute_report_binds_parent_identity_across_verified_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "substitute_parent_identity"
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path)
    payload = _valid_substitute_report(run_id)
    payload["parent"]["smiles"] = "CCN"  # type: ignore[index]
    _write_substitute_bundle(tmp_path, run_id, payload)

    blocked = server._substitute_report(run_id)

    assert blocked is not None
    assert blocked["status"] == "blocked"
    assert "parent identity" in blocked["error"]


def test_substitute_report_blocks_inconsistent_selection_track(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "invalid_substitute_track"
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path)
    payload = _valid_substitute_report(run_id)
    payload["candidates"][0]["selection_tracks"] = ["cosmetic_material"]  # type: ignore[index]
    _write_substitute_bundle(tmp_path, run_id, payload)

    blocked = server._substitute_report(run_id)

    assert blocked is not None
    assert blocked["status"] == "blocked"
    assert "selection_tracks" in blocked["error"]


def test_substitute_report_requires_hash_bound_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "missing_substitute_manifest"
    report_path = (
        tmp_path
        / run_id
        / server.SUBSTITUTE_RELATIVE_DIR
        / "substitute_report.json"
    )
    report_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps(_valid_substitute_report(run_id)))
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path)

    blocked = server._substitute_report(run_id)

    assert blocked is not None
    assert blocked["status"] == "blocked"
    assert "manifest" in blocked["error"]


def test_substitute_report_blocks_artifact_tampering_after_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "tampered_substitute_report"
    payload = _valid_substitute_report(run_id)
    report_path = _write_substitute_bundle(tmp_path, run_id, payload)
    report_path.write_text(json.dumps({**payload, "claimable": True}))
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path)

    blocked = server._substitute_report(run_id)

    assert blocked is not None
    assert blocked["status"] == "blocked"
    assert "fingerprint mismatch" in blocked["error"]


def test_substitute_report_blocks_retention_without_comparison_strata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "invalid_retention_evidence"
    payload = _valid_substitute_report(run_id)
    payload["candidates"][0]["binding_comparison_strata"] = []  # type: ignore[index]
    _write_substitute_bundle(tmp_path, run_id, payload)
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path)

    blocked = server._substitute_report(run_id)

    assert blocked is not None
    assert blocked["status"] == "blocked"
    assert "binding_comparison" in blocked["error"]


def test_summary_card_preserves_substitute_intent_before_report_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path)

    card = server._summary_card(
        "substitute_running",
        {"preset": "target-id", "mode": "fast"},
        "running",
        analysis_kind="substitute",
    )

    assert card["preset"] == "substitute"
    assert card["analysis_kind"] == "substitute"


def test_workbench_beginner_api_hides_stage0_but_allows_discovery(
    isolated_runtime: DurableCoordinator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="Stage 0"):
        server._start_run({
            "input_type": "smiles",
            "smiles": "CCO",
            "preset": "stage0",
            "mode": "fast",
        })
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        server,
        "get_status",
        lambda force=False: {
            "analysis_readiness": {
                "target_fast": {"ready": True, "missing": []},
            }
        },
    )
    monkeypatch.setattr(server, "_active_job_for_run", lambda _run_id: None)
    monkeypatch.setattr(server, "_require_discovery_alias_package", lambda: None)

    def fake_start(**kwargs: object) -> server.Job:
        captured.update(kwargs)
        return server.Job(
            job_id="discovery-job",
            kind="run",
            run_id=str(kwargs["run_id"]),
        )

    monkeypatch.setattr(server, "_start_process_job", fake_start)
    job = server._start_run({
        "input_type": "smiles",
        "smiles": "CCO",
        "run_id": "discovery_case",
        "preset": "target-id",
        "mode": "fast",
        "evidence_mode": "discovery",
    })

    assert job.job_id == "discovery-job"
    command = captured["command"]
    assert isinstance(command, list)
    assert command[command.index("--evidence-mode") + 1] == "discovery"
    payload_extra = captured["payload_extra"]
    assert isinstance(payload_extra, dict)
    assert payload_extra["run_profile"]["evidence_mode"] == "discovery"


def test_workbench_discovery_requires_valid_stage0_alias_package(
    isolated_runtime: DurableCoordinator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        server,
        "get_status",
        lambda force=False: {
            "analysis_readiness": {
                "target_fast": {"ready": True, "missing": []},
            }
        },
    )

    def reject_aliases() -> None:
        raise RuntimeError("Discovery alias package invalid")

    monkeypatch.setattr(server, "_require_discovery_alias_package", reject_aliases)

    with pytest.raises(RuntimeError, match="alias package invalid"):
        server._start_run({
            "input_type": "smiles",
            "smiles": "CCO",
            "run_id": "discovery_missing_aliases",
            "preset": "target-id",
            "mode": "fast",
            "evidence_mode": "discovery",
        })


def test_workbench_discovery_gate_uses_full_stage0_alias_integrity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Path] = []

    def stale_upstream(repo: Path) -> object:
        seen.append(repo)
        return server.stage0_verify.Check(
            "discovery_exact_alias_integrity",
            False,
            "errors=['sources.inputs.ChEMBL.manifest.sha256']",
        )

    monkeypatch.setattr(
        server.stage0_verify,
        "chk_discovery_alias_integrity",
        stale_upstream,
    )

    with pytest.raises(RuntimeError, match="ChEMBL.manifest.sha256"):
        server._require_discovery_alias_package()

    assert seen == [server.ROOT]


def _ready_workbench(monkeypatch, captured: dict) -> None:
    monkeypatch.setattr(
        server,
        "get_status",
        lambda force=False: {
            "analysis_readiness": {"target_fast": {"ready": True, "missing": []}}
        },
    )
    monkeypatch.setattr(server, "_active_job_for_run", lambda _run_id: None)

    def fake_start(**kwargs: object) -> server.Job:
        captured.update(kwargs)
        return server.Job(job_id="gate-job", kind="run", run_id=str(kwargs["run_id"]))

    monkeypatch.setattr(server, "_start_process_job", fake_start)


TRIPEPTIDE = "CC(C)C[C@@H](N)C(=O)N[C@@H](C)C(=O)N[C@@H](Cc1ccccc1)C(=O)O"
OXYBENZONE = "COc1ccc(C(=O)c2ccccc2)c(O)c1"


def test_the_browser_refuses_a_smiles_outside_the_documented_scope(monkeypatch) -> None:
    captured: dict = {}
    _ready_workbench(monkeypatch, captured)

    with pytest.raises(ValueError, match="적용 범위 밖"):
        server._start_run({
            "input_type": "smiles",
            "smiles": TRIPEPTIDE,
            "run_id": "gate_smiles_case",
            "preset": "target-id",
            "mode": "fast",
        })

    assert captured == {}, "a refused input must not start a job"


def test_the_browser_refuses_a_uv_filter(monkeypatch) -> None:
    captured: dict = {}
    _ready_workbench(monkeypatch, captured)

    with pytest.raises(ValueError, match="자외선차단"):
        server._start_run({
            "input_type": "smiles",
            "smiles": OXYBENZONE,
            "run_id": "gate_uv_case",
            "preset": "target-id",
            "mode": "fast",
        })


def test_the_browser_refuses_an_out_of_scope_sdf_upload(monkeypatch) -> None:
    """SDF uploads walked straight past the gate that covered the SMILES field."""
    captured: dict = {}
    _ready_workbench(monkeypatch, captured)
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(TRIPEPTIDE)
    AllChem.Compute2DCoords(mol)
    sdf_text = Chem.MolToMolBlock(mol) + "$$$$\n"

    with pytest.raises(ValueError, match="적용 범위 밖"):
        server._start_run({
            "input_type": "sdf",
            "sdf_content": sdf_text,
            "run_id": "gate_sdf_case",
            "preset": "target-id",
            "mode": "fast",
        })

    assert captured == {}
    assert not (server.UPLOAD_DIR / "gate_sdf_case.sdf").exists(), (
        "a refused upload must not be left on disk"
    )


def test_the_browser_still_accepts_an_sdf_within_scope(monkeypatch) -> None:
    captured: dict = {}
    _ready_workbench(monkeypatch, captured)
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles("Cn1c(=O)c2c(ncn2C)n(C)c1=O")
    AllChem.Compute2DCoords(mol)
    sdf_text = Chem.MolToMolBlock(mol) + "$$$$\n"

    job = server._start_run({
        "input_type": "sdf",
        "sdf_content": sdf_text,
        "run_id": "gate_sdf_ok_case",
        "preset": "target-id",
        "mode": "fast",
    })

    assert job.job_id == "gate-job"
    command = captured["command"]
    assert "--sdf" in command
    server.UPLOAD_DIR.joinpath("gate_sdf_ok_case.sdf").unlink(missing_ok=True)


# --- C-01: a run that failed verification must not read as a success ---


def _write_run_pair(
    runs_dir: Path, run_id: str, *, verification: dict | None
) -> None:
    run = runs_dir / run_id
    run.mkdir(parents=True)
    (run / "run_summary.json").write_text(
        json.dumps({
            "preset": "target-id",
            "mode": "fast",
            "compound": {"canonical_smiles": "CCO"},
            "overall_decision": {"decision": "PASS", "claimable": True},
        }),
        encoding="utf-8",
    )
    if verification is not None:
        (run / "run_verification.json").write_text(
            json.dumps(verification), encoding="utf-8"
        )


def test_a_failed_verification_is_not_reported_as_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`run_summary.json` is written before the verifier runs.

    Treating its existence as success let a run the verifier rejected present
    itself as completed and claimable. Six of the 44 committed runs are in that
    state.
    """
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path)
    _write_run_pair(
        tmp_path,
        "rejected",
        verification={
            "status": "failed",
            "verifier_status": "failed",
            "verifier_contract_error": "verifier JSON status was not ok: failed",
        },
    )

    card = server._summary_card("rejected", server._run_summary("rejected"), "completed")

    assert card["status"] == "verification_failed"
    assert card["claimable"] is False
    assert card["verification"]["ok"] is False
    assert "not ok" in card["verification"]["error"]


def test_a_run_with_no_verification_record_is_not_claimable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path)
    _write_run_pair(tmp_path, "unchecked", verification=None)

    card = server._summary_card("unchecked", server._run_summary("unchecked"), "completed")

    assert card["status"] == "unverified"
    assert card["claimable"] is False
    assert card["verification"]["checked"] is False


def test_a_verified_run_still_reports_completed_and_claimable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path)
    _write_run_pair(
        tmp_path, "clean", verification={"status": "ok", "verifier_status": "ok"}
    )

    card = server._summary_card("clean", server._run_summary("clean"), "completed")

    assert card["status"] == "completed"
    assert card["claimable"] is True


def test_a_running_job_keeps_its_lifecycle_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verification only refines a finished run, never an in-flight one."""
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path)
    _write_run_pair(tmp_path, "inflight", verification=None)

    card = server._summary_card("inflight", server._run_summary("inflight"), "running")

    assert card["status"] == "running"


# --- M-07 / M-10: progress and rank must describe what actually happened ---


def test_progress_is_read_from_artifacts_not_guessed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every running job used to sit on stage 1 for its whole life."""
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path)
    run = tmp_path / "partway"
    (run / "01_input").mkdir(parents=True)
    (run / "01_input" / "ligand.pdbqt").write_text("x", encoding="utf-8")
    (run / "03_targets").mkdir()
    (run / "03_targets" / "top50.csv").write_text("target_id\n", encoding="utf-8")

    progress = server._run_progress("partway")

    done = {stage["key"] for stage in progress["stages"] if stage["done"]}
    assert done == {"input", "targets"}
    assert progress["completed"] == 2
    assert progress["determinate"] is True


def test_progress_says_it_does_not_know_rather_than_inventing_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path)
    (tmp_path / "just_started").mkdir()

    progress = server._run_progress("just_started")

    assert progress["completed"] == 0
    assert progress["determinate"] is False
    assert all(stage["done"] is False for stage in progress["stages"])


def test_an_empty_stage_directory_does_not_count_as_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path)
    run = tmp_path / "hollow"
    (run / "03_targets").mkdir(parents=True)

    progress = server._run_progress("hollow")

    assert progress["determinate"] is False


def test_filtering_keeps_each_targets_place_in_the_full_ranking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A globally 3rd target used to display as rank 1 once a filter narrowed the view."""
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path)
    run = tmp_path / "ranked"
    (run / "03_targets").mkdir(parents=True)
    (run / "03_targets" / "ranked_targets_v3_with_efficacy.csv").write_text(
        "target_id,final_score,skin_score,skin_tier,docking_rrf,source_count,sources\n"
        "P00001,0.9,0.9,very_high,0.5,2,autodock;gnina\n"
        "P00002,0.8,0.8,high,0.4,2,autodock;gnina\n"
        "P00003,0.7,0.1,very_low,0.3,2,autodock;gnina\n",
        encoding="utf-8",
    )

    summary = {"preset": "target-id", "mode": "fast"}
    payload = server._target_rows("ranked", summary, skin_tier="very_low")

    assert len(payload["rows"]) == 1
    row = payload["rows"][0]
    assert row["rank"] == 3, "the position in the full ranking must survive filtering"
    assert row["original_rank"] == 3
    assert row["filtered_position"] == 1


def test_a_missing_viewer_explains_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The viewer is the reader's 3D view; its absence needs a reason.

    The builder's failure went to stderr, which in a Workbench-launched run is
    buried in a 48 KB log tail, so the viewer simply was not there.
    """
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path)
    run = tmp_path / "no_viewer"
    run.mkdir()
    (run / "run_viewer_status.json").write_text(
        json.dumps({"status": "failed", "reason": "채점 결과 CSV를 찾지 못했습니다."}),
        encoding="utf-8",
    )

    state = server._viewer_state("no_viewer")

    assert state["available"] is False
    assert "채점 결과 CSV" in state["reason"]


def test_a_built_viewer_is_offered_to_the_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path)
    viewer = tmp_path / "with_viewer" / "viewer"
    viewer.mkdir(parents=True)
    (viewer / "index.html").write_text("<html></html>", encoding="utf-8")

    state = server._viewer_state("with_viewer")

    assert state["available"] is True
    assert state["path"] == "viewer/index.html"


# --- Downloads and the embedded viewer ---


def test_a_run_file_can_be_downloaded_without_being_promoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing in production calls the promote API.

    Every artifact row therefore rendered as "등록 필요" with no link, and a
    reader could not get a single file out of the browser.
    """
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path)
    run = tmp_path / "downloadable"
    run.mkdir()
    (run / "run_summary.md").write_text("# summary\n", encoding="utf-8")

    row = server._run_file_row("downloadable", "run_summary_markdown", "run_summary.md")

    assert row["status"] == "available"
    assert row["size_bytes"] == len("# summary\n")
    assert row["download_url"].endswith("path=run_summary.md")
    assert server._run_file_for_download("downloadable", "run_summary.md") is not None


@pytest.mark.parametrize(
    "hostile",
    [
        "../../../etc/passwd",
        "/etc/passwd",
        "viewer/../../../../etc/passwd",
        "..\\\\windows\\\\system32",
        "",
    ],
)
def test_a_download_path_cannot_leave_the_run_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hostile: str
) -> None:
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path)
    (tmp_path / "guarded").mkdir()

    assert server._run_file_for_download("guarded", hostile) is None


def test_a_symlink_out_of_the_run_directory_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path)
    run = tmp_path / "linked"
    run.mkdir()
    secret = tmp_path / "outside.json"
    secret.write_text("{}", encoding="utf-8")
    (run / "escape.json").symlink_to(secret)

    assert server._run_file_for_download("linked", "escape.json") is None


def test_only_readable_result_types_are_served(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The endpoint must not become a general file server for the run dir."""
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path)
    run = tmp_path / "typed"
    run.mkdir()
    (run / "notes.csv").write_text("a,b\n", encoding="utf-8")
    (run / "core.dump").write_text("binary", encoding="utf-8")

    assert server._run_file_for_download("typed", "notes.csv") is not None
    assert server._run_file_for_download("typed", "core.dump") is None


def test_the_bundle_carries_the_summary_and_the_viewer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import io
    import zipfile

    monkeypatch.setattr(server, "RUNS_DIR", tmp_path)
    run = tmp_path / "bundled"
    (run / "viewer" / "targets").mkdir(parents=True)
    (run / "run_summary.md").write_text("# summary\n", encoding="utf-8")
    (run / "viewer" / "index.html").write_text("<html></html>", encoding="utf-8")
    (run / "viewer" / "targets" / "P00001.html").write_text("<html></html>", encoding="utf-8")

    payload = server._run_bundle_zip("bundled")

    assert payload is not None
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = set(archive.namelist())
    assert "bundled/run_summary.md" in names
    assert "bundled/viewer/index.html" in names
    assert "bundled/viewer/targets/P00001.html" in names


def test_the_viewer_may_be_framed_by_the_workbench_but_nothing_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both halves are needed: the child allows the parent, the parent allows the frame.

    frame-ancestors alone was not enough - the Workbench's own CSP had
    default-src 'none' with no frame-src, so the browser blocked the frame.
    """
    source = (ROOT / "workbench" / "server.py").read_text(encoding="utf-8")
    assert "frame-src 'self'" in source
    assert 'frame_ancestors = "\'self\'" if embeddable else "\'none\'"' in source
    assert '"X-Frame-Options", "SAMEORIGIN" if embeddable else "DENY"' in source


@pytest.mark.skipif(
    importlib.util.find_spec("playwright") is None
    or shutil.which("google-chrome") is None,
    reason="Playwright and local Google Chrome are required for the browser gate",
)
def test_the_workbench_renders_the_runs_3d_viewer_in_a_real_browser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The embedded viewer needs three things static checks cannot verify.

    Found by running this: the Workbench's own CSP had no frame-src so the
    frame was blocked outright; Mol* loads its WebAssembly from a data: URI
    which connect-src refused; and re-parenting the frame on the five-second
    poll reloaded it, throwing the reader back to the index every time.
    """
    from playwright.sync_api import sync_playwright

    run_id = "browser_gate_run"
    run_dir = tmp_path / run_id
    (run_dir / "03_targets" / "mode_fast").mkdir(parents=True)
    (run_dir / "run_summary.json").write_text(
        json.dumps({
            "preset": "target-id",
            "mode": "fast",
            "compound": {"canonical_smiles": "CCO"},
            "overall_decision": {"decision": "REVIEW", "claimable": False, "reasons": []},
            "artifacts": {},
        }),
        encoding="utf-8",
    )
    (run_dir / "run_verification.json").write_text(
        json.dumps({"status": "ok", "verifier_status": "ok"}), encoding="utf-8"
    )
    stage = run_dir / "03_targets" / "mode_fast"
    (stage / "top50.csv").write_text(
        "target_id,rrf_score,source_count,sources,autodock_energy_kcal_mol\n"
        "P14679,0.05,1,autodock,-6.10\n",
        encoding="utf-8",
    )
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "P14679_clean.pdb").write_text(_MINIMAL_RECEPTOR_PDB, encoding="utf-8")
    metadata = tmp_path / "proteinatlas.tsv"
    metadata.write_text(
        "Gene\tUniprot\tGene description\tMolecular function\nTYR\tP14679\tTyrosinase\tEnzyme\n",
        encoding="utf-8",
    )

    sys.path.insert(0, str(ROOT / "scripts"))
    from make_results_viewer import build as build_viewer

    build_viewer(run_dir, run_dir / "viewer", metadata, clean, 10)

    monkeypatch.setattr(server, "RUNS_DIR", tmp_path)
    httpd = server.create_server(host="127.0.0.1", port=0)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as engine:
            browser = engine.chromium.launch(channel="chrome", headless=True)
            page = browser.new_page(viewport={"width": 1400, "height": 1000})
            console_errors: list[str] = []
            page.on(
                "console",
                lambda m: console_errors.append(m.text) if m.type == "error" else None,
            )
            page.goto(f"http://127.0.0.1:{port}/", wait_until="networkidle")
            page.wait_for_timeout(2000)
            page.evaluate("() => { location.hash = '#runs'; }")
            page.wait_for_timeout(1500)
            page.click(f'.open-run[data-run-id="{run_id}"]', timeout=15000)
            page.wait_for_selector("#result-viewer iframe.viewer-frame", timeout=15000)
            page.wait_for_timeout(3000)

            downloads = page.eval_on_selector_all(
                ".download-bar a", "els => els.map(e => e.getAttribute('href'))"
            )
            frame = page.frame_locator("iframe.viewer-frame")
            frame.locator("table tbody tr td a").first.click(timeout=20000)
            page.wait_for_timeout(9000)
            target_frames = [f for f in page.frames if "/viewer/targets/" in (f.url or "")]
            loaded = target_frames[0].evaluate(
                """() => {
                  const cells = Array.from(window.__viewerPlugin.state.data.cells.values());
                  let trajectories = 0;
                  for (const cell of cells) {
                    if (cell.obj && cell.obj.type && cell.obj.type.name === 'Trajectory') {
                      trajectories += 1;
                    }
                  }
                  const fallback = document.getElementById('fallback');
                  return {trajectories, fallbackVisible: !fallback.hidden};
                }"""
            ) if target_frames else None
            browser.close()
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)

    assert console_errors == []
    assert any("bundle.zip" in (href or "") for href in downloads)
    assert loaded is not None, "clicking a gene did not open its target page"
    assert loaded["fallbackVisible"] is False
    assert loaded["trajectories"] >= 1


# --- Input preview: what the parser read, before hours of compute ---


def test_the_preview_shows_the_molecule_the_parser_actually_read() -> None:
    preview = server.compound_preview({"smiles": "CC(=O)Oc1ccccc1C(=O)O"})

    assert preview["valid"] is True
    assert preview["canonical_smiles"] == "CC(=O)Oc1ccccc1C(=O)O"
    assert preview["inchikey"] == "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
    assert preview["formula"] == "C9H8O4"
    assert preview["svg"].lstrip().startswith("<?xml") or "<svg" in preview["svg"]
    assert preview["verdict"] == "in_scope"
    assert preview["can_start"] is True


def test_the_preview_surfaces_warnings_the_run_never_showed() -> None:
    """The server computed these on every run and then discarded them."""
    preview = server.compound_preview({"smiles": "CCO"})

    assert preview["verdict"] == "review"
    assert preview["can_start"] is True, "a panel warning is not a refusal"
    details = " ".join(item["detail"] for item in preview["warnings"])
    assert "검증 패널" in details
    assert {item["key"] for item in preview["properties"]} >= {
        "molecular_weight", "logp", "heavy_atoms", "qed"
    }


def test_the_preview_refuses_an_out_of_scope_compound_before_it_runs() -> None:
    peptide = "CC(C)CC(N)C(=O)NC(CC(C)C)C(=O)NC(CC(C)C)C(=O)O"

    preview = server.compound_preview({"smiles": peptide})

    assert preview["verdict"] == "out_of_scope"
    assert preview["can_start"] is False
    assert any(item["code"] == "peptide" for item in preview["exclusions"])
    assert preview["message"]


def test_the_preview_names_an_unreadable_smiles_immediately() -> None:
    preview = server.compound_preview({"smiles": "C1CC1))("})

    assert preview["valid"] is False
    assert preview["can_start"] is False
    assert "읽지 못했습니다" in preview["message"]


@pytest.mark.parametrize("bad", ["", "   "])
def test_the_preview_requires_input(bad: str) -> None:
    with pytest.raises(ValueError):
        server.compound_preview({"smiles": bad})


def test_the_preview_refuses_an_absurdly_long_string() -> None:
    with pytest.raises(ValueError, match="너무 깁니다"):
        server.compound_preview({"smiles": "C" * (server.MAX_SMILES_LENGTH + 1)})


@pytest.mark.skipif(
    importlib.util.find_spec("playwright") is None
    or shutil.which("google-chrome") is None,
    reason="Playwright and local Google Chrome are required for the browser gate",
)
def test_typing_a_peptide_disables_the_start_button_in_the_browser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate has to reach the button, not just the JSON."""
    from playwright.sync_api import sync_playwright

    monkeypatch.setattr(server, "RUNS_DIR", tmp_path)
    httpd = server.create_server(host="127.0.0.1", port=0)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as engine:
            browser = engine.chromium.launch(channel="chrome", headless=True)
            page = browser.new_page(viewport={"width": 1500, "height": 1100})
            console_errors: list[str] = []
            page.on(
                "console",
                lambda m: console_errors.append(m.text) if m.type == "error" else None,
            )
            page.goto(f"http://127.0.0.1:{port}/", wait_until="networkidle")
            page.wait_for_timeout(1500)
            page.evaluate("() => { location.hash = '#analyze'; }")
            page.wait_for_timeout(1200)

            page.fill("#smiles-input", "CC(=O)Oc1ccccc1C(=O)O")
            page.dispatch_event("#smiles-input", "input")
            page.wait_for_timeout(2200)
            drawn = page.eval_on_selector_all("#structure-preview-art svg path", "e => e.length")
            in_scope_verdict = page.text_content("#preview-verdict")

            page.fill("#smiles-input", "CC(C)CC(N)C(=O)NC(CC(C)C)C(=O)NC(CC(C)C)C(=O)O")
            page.dispatch_event("#smiles-input", "input")
            page.wait_for_timeout(2200)
            blocked_verdict = page.text_content("#preview-verdict")
            blocked_submit = page.eval_on_selector("#start-analysis", "e => e.disabled")
            browser.close()
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)

    assert console_errors == []
    assert drawn > 0, "the structure was not drawn"
    assert in_scope_verdict.strip() == "분석 가능"
    assert blocked_verdict.strip() == "분석할 수 없음"
    assert blocked_submit is True


# --- Ingredient-name lookup ---


def test_a_korean_ingredient_name_resolves_to_a_structure() -> None:
    """The reader is a Korean wet-lab researcher; the upstream tables have no Korean."""
    result = server.compound_search("나이아신아마이드")

    assert result["available"] is True
    assert result["matches"], "the curated Korean alias did not resolve"
    top = result["matches"][0]
    assert top["exact"] is True
    assert top["smiles"] == "NC(=O)c1cccnc1"


def test_a_chemical_name_and_an_inci_name_reach_the_same_molecule() -> None:
    inci = server.compound_search("niacinamide")["matches"]
    chemical = server.compound_search("nicotinamide")["matches"]

    assert inci and chemical
    assert inci[0]["smiles"] == chemical[0]["smiles"]


def test_a_partial_name_offers_candidates_without_guessing() -> None:
    result = server.compound_search("레티")

    names = [match["name"] for match in result["matches"]]
    assert "레티놀" in names
    assert all(match["exact"] is False for match in result["matches"])


def test_name_search_ignores_input_too_short_to_mean_anything() -> None:
    assert server.compound_search("a")["matches"] == []
    assert server.compound_search("  ")["matches"] == []


def test_name_search_reports_no_match_rather_than_a_wrong_one() -> None:
    assert server.compound_search("zzzznotathing")["matches"] == []


def test_hangul_survives_name_normalisation() -> None:
    """An ASCII-only character class silently erased every Korean name."""
    assert server._normalise_name("나이아신아마이드") == "나이아신아마이드"
    assert server._normalise_name("Kojic  Acid!") == "kojic acid"


# --- Authentication and the remote bind it unlocks ---


@pytest.fixture
def _auth_off():
    """REQUIRE_AUTH is process-global; leave it as the local default."""
    yield
    server.REQUIRE_AUTH = False


def test_a_remote_bind_without_authentication_is_refused(_auth_off) -> None:
    """Every read endpoint was open, so exposing the port exposed everything."""
    with pytest.raises(ValueError, match="인증이 필요합니다"):
        server.create_server(host="0.0.0.0", port=0, require_auth=False)


def test_a_remote_bind_turns_authentication_on_by_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _auth_off
) -> None:
    """It must not be possible to expose the port unauthenticated by accident."""
    monkeypatch.setattr(server, "ACCESS_TOKEN_PATH", tmp_path / "access.token")
    httpd = server.create_server(host="0.0.0.0", port=0)
    try:
        assert server.REQUIRE_AUTH is True
    finally:
        httpd.server_close()


def test_a_local_bind_keeps_the_single_user_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _auth_off
) -> None:
    monkeypatch.setattr(server, "ACCESS_TOKEN_PATH", tmp_path / "access.token")
    httpd = server.create_server(host="127.0.0.1", port=0)
    try:
        assert server.REQUIRE_AUTH is False
    finally:
        httpd.server_close()


def test_a_session_needs_the_real_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "ACCESS_TOKEN_PATH", tmp_path / "access.token")
    token = server.access_token()

    with pytest.raises(PermissionError):
        server.open_session("not-the-token")
    with pytest.raises(PermissionError):
        server.open_session("")

    session_id = server.open_session(token)
    assert server.session_is_valid(session_id)
    server.close_session(session_id)
    assert not server.session_is_valid(session_id)


def test_an_expired_session_stops_working(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "ACCESS_TOKEN_PATH", tmp_path / "access.token")
    monkeypatch.setattr(server, "SESSION_TTL_SECONDS", -1)

    session_id = server.open_session(server.access_token())

    assert not server.session_is_valid(session_id)


def test_the_access_token_file_is_owner_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "access.token"
    monkeypatch.setattr(server, "ACCESS_TOKEN_PATH", path)

    token = server.access_token()

    assert len(token) >= 32
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert server.access_token() == token, "the token must be stable across calls"

    path.chmod(0o644)
    with pytest.raises(RuntimeError, match="0600"):
        server.access_token()


def test_reads_are_refused_without_a_session_when_auth_is_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _auth_off
) -> None:
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(server, "ACCESS_TOKEN_PATH", tmp_path / "access.token")
    httpd = server.create_server(host="127.0.0.1", port=0, require_auth=True)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    try:
        # The shell and the login route stay reachable so a browser can render
        # the sign-in screen at all.
        for open_path in ("/", "/api/identity", "/api/session"):
            with urlopen(base + open_path, timeout=5) as response:
                assert response.status == 200

        for guarded in ("/api/status", "/api/runs", "/api/setup"):
            with pytest.raises(HTTPError) as error:
                urlopen(base + guarded, timeout=5)
            assert error.value.code == 401, guarded
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


# --- Batch runs: one compound per request made screening impossible ---


def test_a_batch_resolves_names_and_smiles_together() -> None:
    items = server._parse_batch_items({
        "compounds": "나이아신아마이드\nCC(=O)Oc1ccccc1C(=O)O\n아스피린,CC(=O)Oc1ccccc1C(=O)O\nzzz-not-a-thing"
    })

    labels = [(item.label, item.status) for item in items]
    assert labels[0] == ("나이아신아마이드", "queued")
    assert items[0].smiles == "NC(=O)c1cccnc1"
    assert items[1].status == "queued"
    # The same structure twice is a mistake, not two experiments.
    assert items[2].status == "skipped" and "이미 목록에" in items[2].error
    assert items[3].status == "skipped" and "찾지 못했" in items[3].error


def test_a_batch_accepts_a_list_of_objects_or_bare_strings() -> None:
    from_objects = server._parse_batch_items({
        "compounds": [{"label": "카페인", "smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O"}]
    })
    from_strings = server._parse_batch_items({"compounds": ["Caffeine"]})

    assert from_objects[0].smiles == from_strings[0].smiles


@pytest.mark.parametrize("payload", [{"compounds": ""}, {"compounds": []}, {}])
def test_an_empty_batch_is_refused(payload: dict) -> None:
    with pytest.raises(ValueError, match="실행할 화합물이 없습니다"):
        server._parse_batch_items(payload)


def test_a_batch_is_capped() -> None:
    with pytest.raises(ValueError, match="최대 50개"):
        server._parse_batch_items({"compounds": "\n".join(["CCO"] * 60)})


def test_a_batch_runs_its_compounds_one_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """They contend for the same GPU, so sequential is the only correct order."""
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(server, "BATCH_POLL_SECONDS", 0.01)
    started: list[str] = []
    concurrent: list[int] = []
    live = {"count": 0}

    def fake_start(payload: dict) -> server.Job:
        started.append(payload["smiles"])
        live["count"] += 1
        concurrent.append(live["count"])
        run_id = f"run_{len(started)}"
        run = tmp_path / run_id
        run.mkdir()
        (run / "run_summary.json").write_text(
            json.dumps({"overall_decision": {"decision": "REVIEW"}}), encoding="utf-8"
        )
        (run / "run_verification.json").write_text(
            json.dumps({"status": "ok", "verifier_status": "ok"}), encoding="utf-8"
        )
        return server.Job(job_id=run_id, kind="run", run_id=run_id, status="running")

    def fake_active(run_id: str):
        live["count"] = 0
        return None

    monkeypatch.setattr(server, "_start_run", fake_start)
    monkeypatch.setattr(server, "_active_job_for_run", fake_active)

    batch = server.start_batch({
        "compounds": ["CCO", "Cn1c(=O)c2c(ncn2C)n(C)c1=O"],
        "preset": "safety",
        "mode": "fast",
    })
    for _ in range(500):
        if batch.status != "running":
            break
        time.sleep(0.02)

    assert batch.status == "completed"
    assert len(started) == 2
    assert max(concurrent) == 1, "two runs were in flight at once"
    assert [item.status for item in batch.items] == ["completed", "completed"]


def test_a_batch_records_why_an_item_did_not_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(server, "BATCH_POLL_SECONDS", 0.01)

    def refuse(payload: dict) -> server.Job:
        raise ValueError("GPU가 이미 사용 중입니다.")

    monkeypatch.setattr(server, "_start_run", refuse)
    monkeypatch.setattr(server, "_active_job_for_run", lambda run_id: None)

    batch = server.start_batch({"compounds": ["CCO"]})
    for _ in range(500):
        if batch.status != "running":
            break
        time.sleep(0.02)

    assert batch.items[0].status == "failed"
    assert "GPU" in batch.items[0].error


def test_a_batch_can_be_cancelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path)
    batch = server.Batch(
        batch_id="batch_test", items=[server.BatchItem("a", "CCO")],
        preset="safety", mode="fast", evidence_mode="evidence", created_at="now",
    )
    with server._BATCHES_LOCK:
        server._BATCHES[batch.batch_id] = batch

    cancelled = server.cancel_batch("batch_test")

    assert cancelled is not None
    assert cancelled.cancel_requested is True
    assert cancelled.items[0].status == "cancelled"
    assert server.cancel_batch("batch_missing") is None


# --- Vendored structure editor ---


def test_the_structure_editor_is_vendored_with_its_licence() -> None:
    """It ships in the repo so the Workbench needs no network to draw a molecule."""
    vendor = ROOT / "workbench" / "static" / "vendor" / "jsme"

    assert (vendor / "LICENSE").is_file()
    licence = (vendor / "LICENSE").read_text(encoding="utf-8")
    assert "Redistribution and use in source and binary forms" in licence
    assert (vendor / "jsme" / "jsme.nocache.js").is_file()
    assert (vendor / "editor.html").is_file()
    assert (vendor / "editor.js").is_file()


def test_the_editor_page_carries_no_inline_script_of_our_own() -> None:
    """GWT's own bootstrap needs 'unsafe-inline'; ours should not add to it."""
    page = (ROOT / "workbench" / "static" / "vendor" / "jsme" / "editor.html").read_text(
        encoding="utf-8"
    )
    for match in re.finditer(r"<script([^>]*)>", page):
        assert "src=" in match.group(1), f"inline script in editor.html: {match.group(0)}"


def test_only_the_editor_frame_relaxes_the_content_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The relaxation must not reach the app page.

    JSME cannot run under the Workbench policy, so it lives in its own frame.
    That frame holds no run data and its only channel out is a postMessage.
    """
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path)
    httpd = server.create_server(host="127.0.0.1", port=0)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    try:
        with urlopen(base + "/", timeout=5) as response:
            app_policy = response.headers["Content-Security-Policy"]
        with urlopen(base + "/assets/vendor/jsme/editor.html", timeout=5) as response:
            editor_policy = response.headers["Content-Security-Policy"]
            assert response.headers["X-Frame-Options"] == "SAMEORIGIN"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)

    assert "'unsafe-inline'" not in app_policy.split("style-src")[0]
    assert "'unsafe-eval'" not in app_policy
    assert "'unsafe-inline'" in editor_policy.split("style-src")[0]
    assert "'unsafe-eval'" in editor_policy
    # The frame must not be able to navigate anywhere or post a form out.
    assert "form-action 'none'" in editor_policy
    assert "base-uri 'none'" in editor_policy


def test_static_serving_handles_the_editors_binary_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The editor ships images; the static route used to read text only."""
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path)
    httpd = server.create_server(host="127.0.0.1", port=0)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{port}/assets/vendor/jsme/jsme/clear.cache.gif"
        with urlopen(url, timeout=5) as response:
            assert response.status == 200
            assert response.read().startswith(b"GIF")
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


# --- What a skin researcher needs on screen, not one click away ---


def test_the_analysis_choices_are_stated_in_time_and_plain_language() -> None:
    """"Stage0/report-fast 필요" and "Demo:" are not choosable by a biologist."""
    markup = (ROOT / "workbench" / "static" / "index.html").read_text(encoding="utf-8")
    cards = markup.split('name="preset"')

    assert "Stage0/report-fast 필요" not in markup
    assert "Demo: 리간드 유사도" not in markup
    # Every choice says how long it takes.
    for fragment in cards[1:]:
        meta = fragment.split('choice-card-meta">', 1)[1].split("<", 1)[0]
        assert any(unit in meta for unit in ("분", "시간", "일")), meta
    # The report path has never completed; the card has to say so.
    assert "미검증" in markup


def test_the_evidence_mode_switch_is_folded_away() -> None:
    """It is an audit switch. A reader cannot decide it and should not have to."""
    markup = (ROOT / "workbench" / "static" / "index.html").read_text(encoding="utf-8")

    assert 'id="evidence-mode-section"' in markup
    section = markup.split('id="evidence-mode-section"', 1)[1]
    assert section.lstrip().startswith(">") or "advanced-summary" in section[:400]
    assert "기본값 그대로 두면 됩니다" in markup


def test_the_target_table_carries_the_number_that_predicts_recovery() -> None:
    """Nearest-analog similarity decided recovery in the panel measurement.

    It was rendered only inside the per-target drawer, so the column a reader
    scans said nothing about whether a row was worth acting on.
    """
    markup = (ROOT / "workbench" / "static" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "workbench" / "static" / "app.js").read_text(encoding="utf-8")

    assert "근거 유사도" in markup
    assert "function similarityCell(" in script
    assert "${similarityCell(row)}" in script
    # The measured bands, and a self-match must never look like a prediction.
    assert "min: 0.6" in script and "min: 0.4" in script
    assert "daina_is_self_match" in script.split("function similarityCell(", 1)[1][:600]


def test_the_artifact_file_list_no_longer_dominates_the_page() -> None:
    script = (ROOT / "workbench" / "static" / "app.js").read_text(encoding="utf-8")

    assert "artifact-details" in script
    assert "원본 수치를 직접 볼 때만 필요합니다" in script


def test_the_results_lead_with_what_the_compound_does_in_skin() -> None:
    """A skin researcher thinks in 미백/주름/진정, not UniProt accessions."""
    script = (ROOT / "workbench" / "static" / "app.js").read_text(encoding="utf-8")

    assert "function efficacySummaryHtml(" in script
    assert "피부에서 보고된 작용" in script
    for key in ("anti_aging", "whitening", "soothing"):
        assert key in script
    # And it must not be read as proof the compound has that effect.
    assert "이 화합물이 그 효능을 낸다는 근거는 아닙니다" in script


def test_the_three_remote_safety_models_are_disclosed_where_they_matter() -> None:
    """The sensitisation models POST the input structure to third parties.

    HuSSPred, Pred-Skin and StopTox are web APIs at UNC and LabMol. Nothing
    said so, while the README repeatedly emphasised offline operation and the
    upload hint read "로컬 파일만 전송합니다" - which a reader would take to mean
    nothing leaves the machine. For an unpublished cosmetic ingredient that is
    a confidentiality question, not a footnote.
    """
    markup = (ROOT / "workbench" / "static" / "index.html").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    guide = (ROOT / "docs" / "RESEARCHER_GUIDE.md").read_text(encoding="utf-8")

    # Disclosed on the choice itself and again before the run starts.
    assert "choice-card-warn" in markup
    assert 'id="external-notice"' in markup
    assert "외부" in markup.split("choice-card-warn", 1)[1][:200]

    for host in ("husspred.mml.unc.edu", "predskin.labmol.com.br", "stoptox.mml.unc.edu"):
        assert host in readme, host
        assert host in guide, host

    # And the claim that stays true has to stay stated: everything else is local.
    assert "나머지는 전부 이 컴퓨터 안에서만 돕니다" in readme
    assert "로컬 파일만 전송합니다" not in markup


def test_the_safety_layer_reports_what_the_measurement_found() -> None:
    """It was unmeasured until 2026-08-30; now it is measured and weak.

    The guide has to carry the finding rather than the old "we do not know",
    and must still refuse to promote any single model, because the benchmark
    is public and predates the models.
    """
    guide = (ROOT / "docs" / "RESEARCHER_GUIDE.md").read_text(encoding="utf-8")

    section = guide.split("## 4-B.", 1)[1].split("\n## ", 1)[0]
    assert "처음 측정했고" in section
    # The number that matters to a reader deciding what REVIEW means.
    assert "97%를 flag" in section
    assert "47.3%" in section
    assert "triage" in section
    assert "OECD" in section
    # And it must not read as an endorsement of the model that scored well.
    assert "오염을 배제하지 못했다" in section


def test_the_remote_hosts_in_the_docs_match_the_code() -> None:
    """A disclosure that drifts from the code is worse than none."""
    hosts = set()
    for name in ("stage2_husspred.py", "stage2_pred_skin.py", "stage2_stoptox.py"):
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        hosts |= set(re.findall(r"https://([a-z0-9.-]+)/", source))

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for host in hosts:
        assert host in readme, f"{host} is called but not disclosed"
