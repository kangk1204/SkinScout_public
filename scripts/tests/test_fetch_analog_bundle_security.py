"""감사 C05 회귀 — 데이터 번들이 허용 범위 밖을 설치하거나 반쯤 반영하지 못하게 막는다.

`scripts/fetch_analog_bundle.py` 는 tar 를 푸는 신뢰 경계다. 확인하는 것은
"풀렸는가"가 아니라 **잘못된 것을 풀지 않는가**다:

* 매니페스트가 `data/` 밖(예: `scripts/owned.py`)을 가리키면 거부한다
* 절대경로·`..`·심볼릭 링크 멤버를 거부한다
* 파일별 해시가 어긋나면 기존 설치를 한 바이트도 바꾸지 않는다
* 원격 매니페스트는 별도 채널의 digest 없이는 믿지 않는다
* 정상 번들은 `data/` 아래 정해진 파일을 모두 원자적으로 반영한다
"""

from __future__ import annotations

import functools
import hashlib
import http.server
import io
import json
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

FETCH = ROOT / "scripts" / "fetch_analog_bundle.py"
SCHEMA = "skinscout.analog-bundle.v1"

ALLOWED = (
    "data/cosing/cosing.parquet",
    "data/cosing/admet_cache.parquet",
    "data/similarity_index_202609/manifest.json",
    "data/similarity_index_202609/ligands.parquet",
    "data/similarity_index_202609/fingerprints.npy",
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256(path.read_bytes())


def _build_bundle(
    directory: Path,
    entries: list[tuple[str, str, bytes]],
    *,
    manifest_files: dict[str, dict] | None = None,
    bundle_digest: str | None = None,
) -> tuple[Path, Path]:
    """작은 가짜 번들 하나. 실제 325 MB 를 만들지 않고 계약만 본다.

    entries 는 `("file"|"symlink"|"hardlink", 이름, 내용)` 이다. 매니페스트는
    따로 주지 않으면 일반 파일 항목에서 만든다 - 해시 불일치 시험은
    `manifest_files` 로 어긋난 값을 넣는다.
    """
    directory.mkdir(parents=True, exist_ok=True)
    bundle = directory / "skinscout-analog-bundle.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for kind, name, payload in entries:
            info = tarfile.TarInfo(name)
            if kind == "file":
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
            elif kind == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = payload.decode()
                tar.addfile(info)
            elif kind == "hardlink":
                info.type = tarfile.LNKTYPE
                info.linkname = payload.decode()
                tar.addfile(info)
            else:
                raise AssertionError(f"unknown entry kind: {kind}")
    if manifest_files is None:
        manifest_files = {
            name: {"sha256": _sha256(payload), "bytes": len(payload)}
            for kind, name, payload in entries
            if kind == "file"
        }
    manifest = directory / "skinscout-analog-bundle.manifest.json"
    manifest.write_text(
        json.dumps({
            "schema_version": SCHEMA,
            "bundle": {
                "name": bundle.name,
                "sha256": bundle_digest or _sha256_file(bundle),
                "bytes": bundle.stat().st_size,
            },
            "files": manifest_files,
        }),
        encoding="utf-8",
    )
    return bundle, manifest


def _run_fetch(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    scripts = cwd / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "fetch_analog_bundle.py").write_bytes(FETCH.read_bytes())
    return subprocess.run(
        [sys.executable, str(scripts / "fetch_analog_bundle.py"), *args],
        cwd=cwd, capture_output=True, text=True, check=False,
    )


def _no_staging_leftovers(home: Path) -> None:
    leftovers = [p.name for p in home.iterdir() if p.name.startswith(".skinscout-bundle-")]
    assert leftovers == [], f"임시 디렉터리가 남았습니다: {leftovers}"


def test_a_manifest_member_outside_data_is_rejected_and_not_installed(tmp_path: Path) -> None:
    """매니페스트에 `scripts/owned.py` 가 있어도 코드가 거부해야 한다."""
    src = tmp_path / "src"
    entries = [
        ("file", ALLOWED[0], b"library"),
        ("file", "scripts/owned.py", b"malicious"),
    ]
    bundle, manifest = _build_bundle(src, entries)
    home = tmp_path / "install"
    result = _run_fetch(home, "--from", str(bundle), "--manifest", str(manifest))
    assert result.returncode != 0
    assert "허용 목록" in (result.stderr + result.stdout)
    assert not (home / "scripts" / "owned.py").exists()
    assert not (home / "data").exists(), "거부했는데 파일이 남았습니다"


def test_a_traversal_member_is_rejected(tmp_path: Path) -> None:
    src = tmp_path / "src"
    entries = [
        ("file", "../escaped.parquet", b"bad"),
        ("file", ALLOWED[0], b"ok"),
    ]
    bundle, manifest = _build_bundle(src, entries)
    home = tmp_path / "install"
    result = _run_fetch(home, "--from", str(bundle), "--manifest", str(manifest))
    assert result.returncode != 0
    assert not (tmp_path / "escaped.parquet").exists()
    assert not (home / "data").exists()


def test_an_absolute_member_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside-owned.parquet"
    entries = [("file", str(outside), b"bad")]
    manifest_files = {str(outside): {"sha256": _sha256(b"bad"), "bytes": 3}}
    bundle, manifest = _build_bundle(
        tmp_path / "src", entries, manifest_files=manifest_files
    )
    home = tmp_path / "install"
    result = _run_fetch(home, "--from", str(bundle), "--manifest", str(manifest))
    assert result.returncode != 0
    assert not outside.exists()
    assert not (home / "data").exists()


def test_a_symlink_member_is_rejected(tmp_path: Path) -> None:
    """이름이 맞아도 링크면 일반 파일이 아니므로 거부한다."""
    src = tmp_path / "src"
    entries = [("symlink", ALLOWED[0], b"/etc/passwd")]
    manifest_files = {ALLOWED[0]: {"sha256": _sha256(b"/etc/passwd"), "bytes": 0}}
    bundle, manifest = _build_bundle(src, entries, manifest_files=manifest_files)
    home = tmp_path / "install"
    result = _run_fetch(home, "--from", str(bundle), "--manifest", str(manifest))
    assert result.returncode != 0
    assert "링크" in (result.stderr + result.stdout)
    assert not (home / "data").exists()


def test_a_per_file_hash_mismatch_leaves_installed_files_byte_identical(tmp_path: Path) -> None:
    """검증 실패는 기존 설치를 한 바이트도 바꾸면 안 된다."""
    home = tmp_path / "install"
    members = [
        ("file", ALLOWED[0], b"library-v1"),
        ("file", ALLOWED[2], b"index-v1"),
    ]
    good_bundle, good_manifest = _build_bundle(tmp_path / "good", members)
    assert _run_fetch(
        home, "--from", str(good_bundle), "--manifest", str(good_manifest)
    ).returncode == 0
    before = {name: _sha256_file(home / name) for _, name, _ in members}

    # 번들 해시는 실제 tar 와 맞지만, 매니페스트의 파일별 해시만 어긋난다.
    corrupt = [
        ("file", ALLOWED[0], b"library-CORRUPT"),
        ("file", ALLOWED[2], b"index-v1"),
    ]
    expected = {
        name: {"sha256": _sha256(payload), "bytes": len(payload)}
        for _, name, payload in members
    }
    bad_bundle, bad_manifest = _build_bundle(
        tmp_path / "bad", corrupt, manifest_files=expected
    )
    result = _run_fetch(
        home, "--from", str(bad_bundle), "--manifest", str(bad_manifest), "--force"
    )
    assert result.returncode != 0
    assert "해시" in (result.stderr + result.stdout)
    after = {name: _sha256_file(home / name) for name in before}
    assert after == before, "검증 실패가 기존 설치를 바꿨습니다"
    _no_staging_leftovers(home)


def test_an_apply_failure_restores_the_previous_install(tmp_path: Path) -> None:
    """반영 도중 실패해도 먼저 옮긴 파일을 백업에서 되돌려야 한다.

    번들 검증은 전부 통과했는데 두 번째 파일을 옮기다 실패하는 경우를 직접
    만든다 - 기존 파일은 그대로여야 하고 반쯤 반영된 새 파일이 남으면 안 된다.
    """
    import fetch_analog_bundle

    home = tmp_path / "install"
    (home / "data/cosing").mkdir(parents=True)
    (home / ALLOWED[0]).write_bytes(b"old-a")
    (home / ALLOWED[1]).write_bytes(b"old-b")

    staging = tmp_path / "staging"
    (staging / "data/cosing").mkdir(parents=True)
    (staging / ALLOWED[0]).write_bytes(b"new-a")
    # ALLOWED[1] 은 staging 에 없다 - 옮기다 실패한다.
    records = {
        ALLOWED[0]: {"sha256": _sha256(b"new-a"), "bytes": len(b"new-a")},
        ALLOWED[1]: {"sha256": _sha256(b"new-b"), "bytes": len(b"new-b")},
    }
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(fetch_analog_bundle, "ROOT", home)
    try:
        with pytest.raises(FileNotFoundError):
            fetch_analog_bundle._apply_staged(staging, records)
    finally:
        monkeypatch.undo()

    assert (home / ALLOWED[0]).read_bytes() == b"old-a"
    assert (home / ALLOWED[1]).read_bytes() == b"old-b"


def test_a_valid_bundle_applies_all_expected_files_atomically(tmp_path: Path) -> None:
    src = tmp_path / "src"
    payloads = {name: f"payload::{name}".encode() for name in ALLOWED}
    entries = [("file", name, payload) for name, payload in payloads.items()]
    bundle, manifest = _build_bundle(src, entries)
    home = tmp_path / "install"
    result = _run_fetch(
        home, "--from", str(bundle), "--manifest", str(manifest),
        "--manifest-sha256", _sha256_file(manifest),
    )
    assert result.returncode == 0, result.stderr
    for name, payload in payloads.items():
        assert (home / name).read_bytes() == payload
    _no_staging_leftovers(home)


def test_a_manifest_that_does_not_match_the_trusted_digest_is_refused(tmp_path: Path) -> None:
    bundle, manifest = _build_bundle(tmp_path / "src", [("file", ALLOWED[0], b"library")])
    home = tmp_path / "install"
    result = _run_fetch(
        home, "--from", str(bundle), "--manifest", str(manifest),
        "--manifest-sha256", "ab" * 32,
    )
    assert result.returncode != 0
    assert "신뢰된 값" in (result.stderr + result.stdout)
    assert not (home / "data").exists()


def test_a_remote_manifest_is_refused_without_an_out_of_band_digest(tmp_path: Path) -> None:
    """번들과 같은 곳에서 받은 매니페스트는 그 자체로 믿지 않는다."""
    src = tmp_path / "src"
    bundle, manifest = _build_bundle(src, [("file", ALLOWED[0], b"library")])
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(src))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/{manifest.name}"
        home = tmp_path / "install"
        refused = _run_fetch(home, "--from", str(bundle), "--manifest", url)
        assert refused.returncode != 0
        assert "--manifest-sha256" in (refused.stderr + refused.stdout)
        assert not (home / "data").exists()

        accepted = _run_fetch(
            home, "--from", str(bundle), "--manifest", url,
            "--manifest-sha256", _sha256_file(manifest),
        )
        assert accepted.returncode == 0, accepted.stderr
        assert (home / ALLOWED[0]).read_bytes() == b"library"
    finally:
        server.shutdown()
        server.server_close()


def test_the_code_allowlist_matches_the_documented_bundle_members() -> None:
    """코드에 고정한 집합이 빌더가 싣는 다섯 파일과 같아야 한다."""
    import build_analog_bundle
    import fetch_analog_bundle

    assert fetch_analog_bundle.EXPECTED_FILES == frozenset(build_analog_bundle.MEMBERS)


def test_a_manifest_without_bundle_bytes_is_refused(tmp_path: Path) -> None:
    bundle, manifest = _build_bundle(tmp_path / "src", [("file", ALLOWED[0], b"library")])
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    del payload["bundle"]["bytes"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    home = tmp_path / "install"
    result = _run_fetch(home, "--from", str(bundle), "--manifest", str(manifest))

    assert result.returncode != 0
    assert "bytes" in (result.stderr + result.stdout)
    assert not (home / "data").exists()


def test_an_oversized_bundle_response_aborts_at_the_cap_and_preserves_the_install(
    tmp_path: Path,
) -> None:
    src = tmp_path / "src"
    payload = b"library"
    bundle, manifest = _build_bundle(src, [("file", ALLOWED[0], payload)])
    home = tmp_path / "install"
    first = _run_fetch(home, "--from", str(bundle), "--manifest", str(manifest))
    assert first.returncode == 0, first.stderr
    before = _sha256_file(home / ALLOWED[0])

    oversized = src / "served.tar.gz"
    oversized.write_bytes(bundle.read_bytes() + b"overrun")

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(src))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = _run_fetch(
            home,
            "--from", f"http://127.0.0.1:{server.server_address[1]}/{oversized.name}",
            "--manifest", str(manifest),
            "--force",
        )
    finally:
        server.shutdown()
        server.server_close()

    assert result.returncode != 0
    assert "넘었습니다" in (result.stderr + result.stdout)
    assert _sha256_file(home / ALLOWED[0]) == before
    _no_staging_leftovers(home)


def test_a_never_ending_bundle_stream_aborts_under_the_timeout(tmp_path: Path) -> None:
    import fetch_analog_bundle

    class _TrickleHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Length", str(10**9))
            self.end_headers()
            try:
                while True:
                    self.wfile.write(b"x")
                    self.wfile.flush()
                    time.sleep(0.01)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

        def log_message(self, *_args: object) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _TrickleHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    destination = tmp_path / "bundle.tar.gz"
    try:
        with pytest.raises(SystemExit, match="제한 시간"):
            fetch_analog_bundle._retrieve(  # noqa: SLF001
                f"http://127.0.0.1:{server.server_address[1]}/bundle.tar.gz",
                destination,
                expected_bytes=10**9,
                timeout=0.2,
            )
    finally:
        server.shutdown()
        server.server_close()

    assert not destination.exists()
    _no_staging_leftovers(tmp_path)


def test_an_exact_size_remote_bundle_still_installs(tmp_path: Path) -> None:
    src = tmp_path / "src"
    payload = b"library"
    bundle, manifest = _build_bundle(src, [("file", ALLOWED[0], payload)])

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(src))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        home = tmp_path / "install"
        result = _run_fetch(
            home,
            "--from", f"{base}/{bundle.name}",
            "--manifest", f"{base}/{manifest.name}",
            "--manifest-sha256", _sha256_file(manifest),
        )
    finally:
        server.shutdown()
        server.server_close()

    assert result.returncode == 0, result.stderr
    assert (home / ALLOWED[0]).read_bytes() == payload
    _no_staging_leftovers(home)
