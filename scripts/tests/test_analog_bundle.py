"""대체소재 데이터 번들의 계약.

이 번들은 공동연구자 컴퓨터에 **데이터 전체**를 옮기는 수단이다. 원료 구조,
ADMET 예측, 활성 측정 인덱스 - 화면의 세 축이 여기서 온다. 그래서 확인하는 것은
"풀렸는가"가 아니라 **잘못된 것을 풀지 않는가**다:

* 번들이 중간에 바뀌었으면 풀지 않는다
* tar 가 저장소 밖을 가리키면 풀지 않는다
* 매니페스트에 없는 것이 들어 있으면 풀지 않는다
* ADMET 캐시가 라이브러리를 못 덮으면 애초에 묶지 않는다

마지막 것이 특히 그렇다 - 덮지 못한 원료는 화면에서 안전 축이 통째로 중앙값으로
채워지는데, 표는 멀쩡해 보인다. 받는 쪽이 알 방법이 없으므로 보내는 쪽에서 막는다.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

FETCH = ROOT / "scripts" / "fetch_analog_bundle.py"
SCHEMA = "skinscout.analog-bundle.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_bundle(tmp_path: Path, members: dict[str, bytes]) -> tuple[Path, Path]:
    """작은 가짜 번들 하나. 실제 325 MB 를 만들지 않고 계약만 본다."""
    stage = tmp_path / "stage"
    for name, payload in members.items():
        target = stage / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    bundle = tmp_path / "skinscout-analog-bundle.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for name in members:
            tar.add(stage / name, arcname=name)
    manifest = tmp_path / "skinscout-analog-bundle.manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": SCHEMA,
        "bundle": {"name": bundle.name, "sha256": _sha256(bundle),
                   "bytes": bundle.stat().st_size},
        "files": {
            name: {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
            for name, payload in members.items()
        },
    }), encoding="utf-8")
    return bundle, manifest


def _run_fetch(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    scripts = cwd / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "fetch_analog_bundle.py").write_bytes(FETCH.read_bytes())
    return subprocess.run(
        [sys.executable, str(scripts / "fetch_analog_bundle.py"), *args],
        cwd=cwd, capture_output=True, text=True,
    )


MEMBERS = {
    "data/cosing/cosing.parquet": b"library-bytes",
    "data/cosing/admet_cache.parquet": b"admet-bytes",
}


def test_a_matching_bundle_unpacks_where_the_workbench_looks(tmp_path: Path) -> None:
    bundle, manifest = _make_bundle(tmp_path / "src", MEMBERS)
    home = tmp_path / "install"
    result = _run_fetch(home, "--from", str(bundle), "--manifest", str(manifest))
    assert result.returncode == 0, result.stderr
    for name, payload in MEMBERS.items():
        assert (home / name).read_bytes() == payload


def test_a_changed_bundle_is_refused_before_anything_is_written(tmp_path: Path) -> None:
    """해시가 어긋나면 한 바이트도 남기지 않는다."""
    bundle, manifest = _make_bundle(tmp_path / "src", MEMBERS)
    blob = bytearray(bundle.read_bytes())
    blob[len(blob) // 2] ^= 0xFF
    bundle.write_bytes(bytes(blob))

    home = tmp_path / "install"
    result = _run_fetch(home, "--from", str(bundle), "--manifest", str(manifest))
    assert result.returncode != 0
    assert "해시" in (result.stderr + result.stdout)
    assert not (home / "data").exists(), "거부했는데 파일이 남았습니다"


def test_a_member_pointing_outside_the_repository_is_refused(tmp_path: Path) -> None:
    """tar 를 푸는 것은 신뢰 경계다. `..` 로 나가는 항목을 거부해야 한다."""
    bundle, manifest = _make_bundle(tmp_path / "src", MEMBERS)
    # 매니페스트는 그대로 두고 tar 안에만 탈출 항목을 넣는다.
    escaped = tmp_path / "src" / "escaped.tar.gz"
    with tarfile.open(bundle, "r:gz") as source, tarfile.open(escaped, "w:gz") as out:
        for member in source.getmembers():
            out.addfile(member, source.extractfile(member))
        info = tarfile.TarInfo("../escaped.txt")
        info.size = 3
        out.addfile(info, __import__("io").BytesIO(b"bad"))
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["bundle"]["sha256"] = _sha256(escaped)
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    home = tmp_path / "install"
    result = _run_fetch(home, "--from", str(escaped), "--manifest", str(manifest))
    assert result.returncode != 0
    assert not (tmp_path / "escaped.txt").exists()


def test_an_unlisted_member_is_refused(tmp_path: Path) -> None:
    """매니페스트가 모르는 파일이 섞여 있으면 풀지 않는다."""
    extra = dict(MEMBERS, **{"data/cosing/surprise.parquet": b"not-listed"})
    bundle, manifest = _make_bundle(tmp_path / "src", extra)
    listed = json.loads(manifest.read_text(encoding="utf-8"))
    listed["files"].pop("data/cosing/surprise.parquet")
    manifest.write_text(json.dumps(listed), encoding="utf-8")

    home = tmp_path / "install"
    result = _run_fetch(home, "--from", str(bundle), "--manifest", str(manifest))
    assert result.returncode != 0
    assert not (home / "data" / "cosing" / "surprise.parquet").exists()


def test_existing_files_are_not_silently_overwritten(tmp_path: Path) -> None:
    bundle, manifest = _make_bundle(tmp_path / "src", MEMBERS)
    home = tmp_path / "install"
    assert _run_fetch(home, "--from", str(bundle), "--manifest", str(manifest)).returncode == 0
    again = _run_fetch(home, "--from", str(bundle), "--manifest", str(manifest))
    assert again.returncode != 0
    assert "--force" in (again.stderr + again.stdout)


def test_check_mode_notices_a_corrupted_install(tmp_path: Path) -> None:
    bundle, manifest = _make_bundle(tmp_path / "src", MEMBERS)
    home = tmp_path / "install"
    assert _run_fetch(home, "--from", str(bundle), "--manifest", str(manifest)).returncode == 0
    # 길이는 그대로 두고 내용만 바꾼다. 크기 검사로는 안 걸리는 경우라야
    # 해시를 실제로 보는지 확인할 수 있다.
    original = (home / "data/cosing/cosing.parquet").read_bytes()
    (home / "data/cosing/cosing.parquet").write_bytes(bytes(b ^ 0x20 for b in original))
    result = _run_fetch(home, "--check", "--manifest", str(manifest))
    assert result.returncode != 0
    assert "해시가 다름" in (result.stderr + result.stdout)


def test_the_builder_refuses_an_admet_cache_that_does_not_cover_the_library() -> None:
    """묶는 쪽의 게이트. 값이 아니라 **기준**이 맞는지를 본다.

    커버리지는 `cosing.parquet` 의 행이 아니라 화면이 쓰는 라이브러리를 기준으로
    세야 한다. 둘은 다르다 - 라이브러리는 SMILES 를 정규화한 뒤 InChIKey 를 다시
    계산하고 같은 구조를 합쳐 10,120 행이 7,484 종이 된다. 기준을 잘못 잡으면
    제대로 만든 캐시가 74% 로 보이고, 원본 키로 만든 캐시가 100% 로 보인다.
    """
    import inspect

    import build_analog_bundle

    assert build_analog_bundle.MIN_ADMET_COVERAGE >= 0.8
    source = inspect.getsource(build_analog_bundle.admet_coverage)
    assert "load_ingredient_library" in source, (
        "커버리지를 원본 parquet 기준으로 세면 키가 어긋난 캐시를 100% 로 읽는다"
    )
    # 게이트가 묶기 **전에** 걸려야 한다 - 실패한 빌드가 tar 만 남기면 안 된다.
    main_source = inspect.getsource(build_analog_bundle.main)
    assert main_source.index("check_coverage") < main_source.index("tarfile.open")
