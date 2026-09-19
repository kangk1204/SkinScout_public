"""Regression tests for the beginner installation and launch surfaces."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _run(*command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def test_install_scripts_are_valid_bash_and_have_dry_run() -> None:
    scripts = [
        ROOT / "install_skinscout.sh",
        ROOT / "scripts/bootstrap_runtime.sh",
        ROOT / "scripts/conda_host_cli.sh",
    ]
    for script in scripts:
        syntax = _run("bash", "-n", str(script))
        assert syntax.returncode == 0, syntax.stderr

    bootstrap = _run("bash", "scripts/bootstrap_runtime.sh", "--dry-run")
    assert bootstrap.returncode == 0, bootstrap.stderr
    assert "Stage 0" in bootstrap.stdout
    assert "cosmax-base" in bootstrap.stdout

    installer = _run("bash", "install_skinscout.sh", "--dry-run", "--no-launch")
    assert installer.returncode == 0, installer.stderr
    assert "앱 메뉴" in installer.stdout
    assert "--runtime conda" in installer.stdout


def test_installer_runs_as_a_downloaded_standalone_script(tmp_path: Path) -> None:
    standalone = tmp_path / "install_skinscout.sh"
    install_dir = tmp_path / "SkinScout"
    shutil.copy2(ROOT / "install_skinscout.sh", standalone)

    result = subprocess.run(
        [
            "bash",
            str(standalone),
            "--dry-run",
            "--no-launch",
            "--install-dir",
            str(install_dir),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "SkinScout 소스를" in result.stdout
    assert str(install_dir) in result.stdout
    assert "clone 후 scripts/bootstrap_runtime.sh" in result.stdout
    assert "앱 메뉴 바로가기 생성 예정" in result.stdout


def test_readme_quick_start_uses_one_installer_and_real_screenshots() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert readme.index("## Quick Start") < readme.index("## 이 도구가 하려는 것")
    assert "bash <(curl -fsSL https://raw.githubusercontent.com/kangk1204/SkinScout_public/main/install_skinscout.sh)" in readme
    assert "bash ~/SkinScout/install_skinscout.sh --profile full" in readme
    # The screenshots are captured from the running application, in the order a
    # reader meets the screens, by scripts/capture_workbench_screenshots.py.
    assert "돌아가는 프로그램을 그대로 찍은 것" in readme
    assert "capture_workbench_screenshots.py" in readme

    for filename in (
        "wb-1-readiness.png",
        "wb-2-input.png",
        "wb-3-choices.png",
        "wb-4-runs.png",
        "wb-5-results.png",
        "wb-6-viewer.png",
        "wb-7-targets.png",
        "quick-start-results.png",
    ):
        image = ROOT / "docs" / "images" / filename
        assert image.is_file()
        assert image.stat().st_size > 10_000
        assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_conda_host_cli_routes_start_through_installed_environment(
    tmp_path: Path,
) -> None:
    fake_mamba = tmp_path / "micromamba"
    args_path = tmp_path / "args.txt"
    fake_mamba.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"${ARGS_PATH}\"\n",
        encoding="utf-8",
    )
    fake_mamba.chmod(0o755)
    env = os.environ.copy()
    env.update({
        "SKINSCOUT_MICROMAMBA": str(fake_mamba),
        "ARGS_PATH": str(args_path),
    })

    result = subprocess.run(
        ["bash", "scripts/conda_host_cli.sh", "start", "--dry-run", "--port", "18881"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert args_path.read_text().splitlines() == [
        "run",
        "-n",
        "cosmax-base",
        "python",
        str(ROOT / "scripts/start_workbench.py"),
        "--dry-run",
        "--port",
        "18881",
    ]


def test_workbench_launcher_has_non_mutating_dry_run() -> None:
    result = _run(
        sys.executable,
        "scripts/start_workbench.py",
        "--dry-run",
        "--port",
        "18880",
    )
    assert result.returncode == 0, result.stderr
    assert "http://127.0.0.1:18880" in result.stdout


def test_external_model_defaults_are_user_portable() -> None:
    for relative in (
        "psichic/__init__.py",
        "rtmscore/__init__.py",
        "scripts/model_readiness.py",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "/home/keunsoo" not in text
        assert "Path.home()" in text


def test_beginner_ui_explains_readiness_and_result_terms() -> None:
    html = (ROOT / "workbench/static/index.html").read_text(encoding="utf-8")
    javascript = (ROOT / "workbench/static/app.js").read_text(encoding="utf-8")
    assert "무엇이 자동으로 설치되나요?" in html
    assert "analysis-readiness-detail" in html
    assert "결과 용어를 쉽게 설명해 주세요" in javascript
    assert "검증 기반 주장 가능" in javascript
    assert "안전성 분석 가능" in javascript
    assert "검토 후 판단" in javascript
    assert "피부 효능 문헌만 지원" in javascript
    assert "상위 예측 표적은" in javascript
    assert 'claimable === true || decision === "PASS"' not in javascript
    assert "X-SkinScout-Token" in javascript

    server = (ROOT / "workbench/server.py").read_text(encoding="utf-8")
    assert "start_new_session=True" in server
    assert "os.killpg" in server
