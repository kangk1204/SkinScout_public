"""CAS mirror fallback: partial은 미러마다 다시 세고, 잘못된 완성 파일은 버린다.

예산 계산을 루프 밖에서 한 번만 하면 첫 미러의 잘린 응답 뒤에 둘째 미러의 전체
파일이 이어 붙어, 정상 미러가 있어도 다운로드가 실패한다.
"""

from __future__ import annotations

import hashlib
import http.client
import io
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import data_cas  # noqa: E402

CONTENT = b"abcdef"


class _Response(io.BytesIO):
    def __init__(
        self,
        payload: bytes,
        *,
        status: int,
        content_range: str | None = None,
    ) -> None:
        super().__init__(payload)
        self.status = status
        self.headers: dict[str, str] = {}
        if content_range is not None:
            self.headers["Content-Range"] = content_range

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_exc: object) -> bool:
        self.close()
        return False

    def getcode(self) -> int:
        return self.status


class _InterruptedResponse(_Response):
    """첫 read에서 앞 2 byte만 주고, 다음 read에서 연결이 끊긴다."""

    def __init__(self) -> None:
        super().__init__(CONTENT, status=200)
        self._reads = 0

    def read(self, size: int = -1) -> bytes:
        self._reads += 1
        if self._reads == 1:
            return super().read(2)
        raise http.client.IncompleteRead(b"", 4)


def _entry(urls: list[str]) -> dict[str, object]:
    return {
        "path": "models/a.bin",
        "sha256": hashlib.sha256(CONTENT).hexdigest(),
        "size": len(CONTENT),
        "urls": urls,
    }


def _fake_urlopen(responses, requests):
    def fake(request, timeout=60):
        requests.append(request)
        return responses.pop(0)

    return fake


def test_short_first_mirror_falls_back_to_exact_second_mirror(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """첫 미러가 2 byte만 주고 끝나도 정상 미러의 6 byte가 성공해야 한다."""
    entry = _entry(["https://first.test/a.bin", "https://second.test/a.bin"])
    responses = [_Response(b"ab", status=200), _Response(CONTENT, status=200)]
    requests: list[object] = []
    monkeypatch.setattr(
        data_cas.urllib.request,
        "urlopen",
        _fake_urlopen(responses, requests),
    )

    target = data_cas.download_entry(entry, staging=tmp_path / "staging")

    assert target.read_bytes() == CONTENT
    assert not (tmp_path / "staging/models/a.bin.part").exists()
    # 잘못된 크기로 끝난 조각은 재개 대상이 아니다. 둘째 요청에 Range가 없어야
    # 전체 파일이 조각 뒤에 append되지 않는다.
    assert len(requests) == 2
    assert "Range" not in requests[1].headers


def test_wrong_hash_completed_partial_is_not_resumed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """크기가 같아도 hash가 틀린 완성 응답은 다음 미러의 재개 씨앗이 아니다."""
    entry = _entry(["https://first.test/a.bin", "https://second.test/a.bin"])
    responses = [_Response(b"abcdeX", status=200), _Response(CONTENT, status=200)]
    requests: list[object] = []
    monkeypatch.setattr(
        data_cas.urllib.request,
        "urlopen",
        _fake_urlopen(responses, requests),
    )

    target = data_cas.download_entry(entry, staging=tmp_path / "staging")

    assert target.read_bytes() == CONTENT
    assert len(requests) == 2
    assert "Range" not in requests[1].headers


def test_interrupted_mirror_resumes_second_with_recomputed_range(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """도중에 끊긴 조각은 살아 있는 앞부분이므로 둘째 미러가 그 뒤부터 받는다."""
    entry = _entry(["https://first.test/a.bin", "https://second.test/a.bin"])
    responses = [
        _InterruptedResponse(),
        _Response(b"cdef", status=206, content_range="bytes 2-5/6"),
    ]
    requests: list[object] = []
    monkeypatch.setattr(
        data_cas.urllib.request,
        "urlopen",
        _fake_urlopen(responses, requests),
    )

    target = data_cas.download_entry(entry, staging=tmp_path / "staging")

    assert target.read_bytes() == CONTENT
    assert requests[1].headers["Range"] == "bytes=2-"


class _TrickleResponse(io.BytesIO):
    """끝나지 않고 한 바이트씩만 흘리는 응답. 상한 없이 읽으면 스스로 멈춘다."""

    MAX_READS = 40

    def __init__(self) -> None:
        super().__init__(b"")
        self.status = 200
        self.headers: dict[str, str] = {}
        self._reads = 0

    def read(self, size: int = -1) -> bytes:
        self._reads += 1
        if self._reads > self.MAX_READS:
            raise AssertionError("stream was read without any cap or timeout")
        time.sleep(0.05)
        return b"x"

    def __enter__(self) -> "_TrickleResponse":
        return self

    def __exit__(self, *_exc: object) -> bool:
        self.close()
        return False

    def getcode(self) -> int:
        return self.status


class _OverrunResponse(io.BytesIO):
    """상한을 넘긴 뒤 더 읽으려 하면 실패하는 응답."""

    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.status = 200
        self.headers: dict[str, str] = {}
        self._reads = 0

    def read(self, size: int = -1) -> bytes:
        self._reads += 1
        if self._reads > 1:
            raise AssertionError("stream was read past the expected-size cap")
        return super().read(size)

    def __enter__(self) -> "_OverrunResponse":
        return self

    def __exit__(self, *_exc: object) -> bool:
        self.close()
        return False

    def getcode(self) -> int:
        return self.status


def _cas_manifest(version: str, content: bytes) -> dict[str, object]:
    return {
        "schema_version": "skinscout.data_cas.v1",
        "bundle": "skinscout-data-core",
        "version": version,
        "image_compatibility": ["skinscout-app@sha256:test"],
        "signature": {
            "policy": "cosign-keyless-or-key-pinned",
            "certificate_identity": "release@skinscout.example",
            "issuer": "https://token.actions.githubusercontent.com",
            "bundle_path": "manifest.sigstore.json",
            "bundle_sha256": hashlib.sha256(b"verified-bundle").hexdigest(),
        },
        "packs": [
            {
                "name": "core",
                "version": version,
                "files": [
                    {
                        "path": "models/a.bin",
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "size": len(content),
                        "urls": ["https://mirror.test/models/a.bin"],
                        "license_spdx": "CC-BY-4.0",
                        "redistribution": True,
                    }
                ],
            }
        ],
    }


def _cas_verifier(
    manifest_bytes: bytes,
    signature: dict,
    manifest_path: Path | None = None,
) -> None:
    assert signature["certificate_identity"] == "release@skinscout.example"


def test_response_past_expected_size_aborts_and_removes_partial(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """expected+1 바이트는 EOF까지 읽지 않고 즉시 상한에서 중단한다."""
    entry = _entry(["https://only.test/a.bin"])
    monkeypatch.setattr(
        data_cas.urllib.request,
        "urlopen",
        _fake_urlopen([_OverrunResponse(CONTENT + b"X")], []),
    )

    with pytest.raises(data_cas.CasError, match="exceeds expected"):
        data_cas.download_entry(entry, staging=tmp_path / "staging")

    assert not (tmp_path / "staging/models/a.bin.part").exists()
    assert not (tmp_path / "staging/models/a.bin").exists()


def test_resumed_response_past_expected_size_is_removed_at_the_cap(
    tmp_path: Path,
    monkeypatch,
) -> None:
    entry = _entry(["https://only.test/a.bin"])
    partial = tmp_path / "staging/models/a.bin.part"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"ab")
    monkeypatch.setattr(
        data_cas.urllib.request,
        "urlopen",
        _fake_urlopen(
            [_Response(b"cdefg", status=206, content_range="bytes 2-6/7")],
            [],
        ),
    )

    with pytest.raises(data_cas.CasError, match="exceeds expected"):
        data_cas.download_entry(entry, staging=tmp_path / "staging")

    assert not partial.exists()


def test_never_ending_trickle_aborts_under_the_timeout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    entry = _entry(["https://stall.test/a.bin"])
    monkeypatch.setattr(
        data_cas.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _TrickleResponse(),
    )

    with pytest.raises(data_cas.CasError, match="timed out"):
        data_cas.download_entry(
            entry,
            staging=tmp_path / "staging",
            timeout=0.01,
        )


def test_oversized_download_leaves_the_active_pack_untouched(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = tmp_path / "store"
    content_v1 = b"installed-version-one"
    offline = tmp_path / "offline"
    (offline / "models").mkdir(parents=True)
    (offline / "models/a.bin").write_bytes(content_v1)
    active = data_cas.activate(
        data_cas.stage_pack(
            _cas_manifest("1.0.0", content_v1),
            "core",
            store=store,
            offline_dir=offline,
            verifier=_cas_verifier,
        ),
        store=store,
        verifier=_cas_verifier,
    )

    content_v2 = b"never-installed-version-two"
    monkeypatch.setattr(
        data_cas.urllib.request,
        "urlopen",
        _fake_urlopen([_Response(content_v2 + b"X", status=200)], []),
    )

    with pytest.raises(data_cas.CasError, match="exceeds expected"):
        data_cas.stage_pack(
            _cas_manifest("2.0.0", content_v2),
            "core",
            store=store,
            verifier=_cas_verifier,
        )

    assert (store / "current").is_symlink()
    assert (store / "current").resolve() == active
    assert (active / "models/a.bin").read_bytes() == content_v1

