"""Regression tests for the environment specifications themselves.

envs/*.yml는 코드가 아니라서 회귀가 건드리지 않는데, 여기가 틀리면 그 스테이지는
아예 돌지 않는다. 이 저장소에서 실제로 겪은 것만 고정한다:

* `molstar-renderer==0.3.0` 은 PyPI 에 없는 패키지였다. viz 환경이 만들어지지
  않아 stage 9 리포트와 stage 11 그림이 한 번도 생성되지 않았다.
* `gpu4pyscf-cuda12x==1.0.0` 은 없는 버전이었다(배포된 것은 1.0).
* `bioemu==1.0.0` 은 yanked 된 프리릴리스였다.
* `- "--no-deps"` 를 pip 목록에 적었다. conda 는 그 목록을 requirements 파일로
  넘기고 pip 은 그 줄을 **패키지 이름으로** 읽는다:
  `ERROR: Invalid requirement: --no-deps`. 환경 자체가 만들어지지 않는다.

네트워크를 쓰지 않는 검사만 둔다. 실제 설치 가능성은 환경을 만들어 봐야 알지만,
형식이 틀린 것은 여기서 잡을 수 있다.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ENV_DIR = ROOT / "envs"

# requirements 파일이 **줄 단위 옵션으로 받아 주는** 것들. 이것들은 conda 의 pip
# 목록에 적어도 정상 동작한다(2026-09-05 실측: --extra-index-url 로 설치 성공).
REQUIREMENTS_FILE_OPTIONS = {
    "--extra-index-url", "--index-url", "--find-links", "--trusted-host",
    "--pre", "--no-binary", "--only-binary", "--require-hashes",
    "--prefer-binary", "--constraint", "--requirement", "--editable",
}
# 나머지 플래그는 `pip install` 의 명령행 옵션이라 requirements 파일에 못 쓴다.
# pip 은 그 줄을 패키지 이름으로 읽는다: `ERROR: Invalid requirement: --no-deps`.
PIP_OPTION = re.compile(r'^\s*-\s*"?(--[a-z-]+)')


def _pip_entries(path: Path) -> list[tuple[int, str]]:
    entries: list[tuple[int, str]] = []
    inside = False
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if re.match(r"\s*- pip:\s*$", line):
            inside = True
            continue
        if not inside:
            continue
        if line.strip() and not line.startswith("      "):
            inside = False
            continue
        if line.strip().startswith("#") or not line.strip():
            continue
        entries.append((number, line))
    return entries


@pytest.mark.parametrize("spec", sorted(ENV_DIR.glob("*.yml")), ids=lambda p: p.name)
def test_no_pip_option_flags_in_the_pip_list(spec: Path) -> None:
    """`--no-deps` 같은 플래그는 여기 적을 수 없다."""
    offenders = []
    for number, line in _pip_entries(spec):
        match = PIP_OPTION.match(line)
        if match and match.group(1) not in REQUIREMENTS_FILE_OPTIONS:
            offenders.append(f"{spec.name}:{number} {line.strip()}")
    assert not offenders, (
        "conda 는 pip 목록을 requirements 파일로 넘기고 pip 은 이 줄을 패키지 "
        "이름으로 읽습니다: " + "; ".join(offenders)
    )


@pytest.mark.parametrize("spec", sorted(ENV_DIR.glob("*.yml")), ids=lambda p: p.name)
def test_every_pip_entry_names_a_package(spec: Path) -> None:
    """이름과 버전만. 경로나 URL 은 재현되지 않는다."""
    bad = []
    for number, line in _pip_entries(spec):
        token = line.strip().lstrip("- ").strip('"').split("#")[0].strip()
        if not token:
            continue
        if token.startswith("--"):
            continue  # 위 테스트가 따로 본다
        if token.startswith(("/", "./", "../", "http://", "https://", "git+")):
            bad.append(f"{spec.name}:{number} {token}")
    assert not bad, "pip 항목이 패키지 이름이 아닙니다: " + "; ".join(bad)


def test_every_env_declares_a_name() -> None:
    """이름이 없으면 micromamba 가 어디에 만들지 정하지 못한다."""
    for spec in sorted(ENV_DIR.glob("*.yml")):
        first = spec.read_text().splitlines()[0]
        assert first.startswith("name:"), f"{spec.name} 에 name: 이 없습니다"
