"""The README is the hand-off document, so its claims are contracts too.

It carries two sets of pictures: six screenshots of the running program and two
mockups of a UI that does not exist. Mixing them up is the single most damaging
thing this file could do to a first delivery, so the distinction is asserted
rather than trusted to whoever edits next.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"


def _text() -> str:
    return README.read_text(encoding="utf-8")


def test_every_local_link_and_image_resolves() -> None:
    """A delivery README with a 404 in it is worse than one without the link."""
    text = _text()
    missing = []
    for match in re.finditer(r'(?:src="|\]\(\./)([^")#]+)', text):
        raw = match.group(1)
        if raw.startswith(("http://", "https://", "mailto:")):
            continue
        target = ROOT / raw.replace("%20", " ")
        if target.suffix and not target.exists():
            missing.append(raw)
    assert not missing, missing


def test_the_real_screenshots_are_still_claimed_as_real() -> None:
    text = _text()

    assert "목업이 아니라" in text
    for name in (
        "wb-1-readiness", "wb-2-input", "wb-3-choices",
        "wb-4-runs", "wb-5-results", "wb-6-viewer",
    ):
        assert f"docs/images/{name}.png" in text, name
    # And they are reproducible, not hand-picked.
    assert "capture_workbench_screenshots.py" in text


def test_the_mockups_are_labelled_as_not_implemented() -> None:
    """They are dark-themed and look finished. Nothing about the image says it
    is a proposal, so the surrounding text has to."""
    text = _text()

    import re

    section = text.split("### 다음 화면 시안", 1)[1].split("### ", 1)[0]
    assert "아직 구현 안 됐습니다" in text.split("### 다음 화면 시안", 1)[1][:40]

    # Matched by meaning, not by sentence: the wording gets edited, and a test
    # that pins the exact phrasing breaks on a rewrite while a deleted
    # disclaimer would slip through some other rephrasing.
    flat = re.sub(r"\s+", " ", section)
    assert re.search(r"코드[는가] 아직 없습니다", flat), "must say the code does not exist"
    # The mobile mockup implies an app that does not exist.
    assert re.search(r"모바일 앱이[^.]{0,20}없습니다", flat), "must deny the mobile app"
    # And it says how to actually view them, since GitHub will not render HTML.
    assert re.search(r"GitHub에서는[^.]{0,12}소스로만 보입니다", flat)
    assert "support.js" in section


def test_the_pipeline_section_matches_the_configured_pipeline() -> None:
    """The numbers a reader uses to judge the tool come from config, so they
    have to be read from it rather than remembered."""
    import json

    import yaml

    text = _text()
    docking = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text(encoding="utf-8"))[
        "docking"
    ]

    assert str(docking["fast_mode_daina_top_n"]) in text
    assert f"{docking['fast_mode_rerank_band']}위" in text or "11~50위" in text

    recipe_path = ROOT / docking["daina_recipe_path"]
    if recipe_path.exists():
        promoted = json.loads(recipe_path.read_text(encoding="utf-8"))
        assert promoted["selected_recipe"]["recipe_id"] in text

    index_manifest = ROOT / docking["daina_recipe_index_dir"] / "manifest.json"
    if index_manifest.exists():
        targets = json.loads(index_manifest.read_text(encoding="utf-8"))["projection"]["targets"]
        assert f"{targets:,}" in text, targets


def test_the_roadmap_does_not_still_promise_what_shipped() -> None:
    """BindingDB sat in "앞으로 할 것" after it was merged and switched on."""
    text = _text()

    roadmap = text.split("### 앞으로 할 것", 1)[1].split("### ", 1)[0]
    assert "BindingDB·GtoPdb를 검색에 합치기" not in roadmap
    assert "1번은 끝났습니다" in roadmap
    assert "248개" not in text, "the pre-measurement estimate; the measured figure is +214"


def test_installation_and_remote_access_are_not_overpromised() -> None:
    text = _text()
    assert "깨끗한 Ubuntu에서 설치부터 전체 분석까지 검증된 배포판을 뜻하지는 않습니다" in text
    assert "Linux·GPU 담당자가 필요합니다" in text
    assert "`--profile full` 한 줄입니다" not in text
    assert "통신이 암호화되지는 않습니다" in text
    assert "SSH 터널" in text


def test_base_pip_phase_preserves_condas_existing_yaml_dependency_bound() -> None:
    import yaml

    spec = yaml.safe_load((ROOT / "envs/base.yml").read_text())
    pip_requirements = next(entry["pip"] for entry in spec["dependencies"] if isinstance(entry, dict))
    assert "ruamel.yaml>=0.17.11,<0.19" in pip_requirements


def test_the_target_first_tool_is_documented_and_runnable() -> None:
    """README에 소개한 명령이 도구의 실제 CLI와 어긋나면 초보자는 거기서 막힌다.

    상세 사용법은 `docs/EXPLORE_TARGET.md`로 옮겼다. README에는 진입점과 링크만
    두고, 옵션 설명은 문서 쪽에서 실제 CLI와 대조한다.
    """
    import subprocess
    import sys

    readme = _text()
    assert "### 표적 단백질부터 시작하기" in readme, "표적 우선 도구 소개가 사라졌습니다"
    assert "scripts/explore_target.py" in readme
    assert "docs/EXPLORE_TARGET.md" in readme

    guide = ROOT / "docs" / "EXPLORE_TARGET.md"
    assert guide.is_file()
    body = guide.read_text(encoding="utf-8")

    script = ROOT / "scripts" / "explore_target.py"
    help_text = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True, text=True, check=True,
    ).stdout
    for flag in ("--search", "--smiles-out", "--print-run-cmds"):
        assert flag in body, f"EXPLORE_TARGET.md에 {flag} 안내가 없습니다"
        assert flag in help_text, f"explore_target.py에 {flag} 옵션이 없습니다"

    assert "run_skinscout.py --smiles" in body
