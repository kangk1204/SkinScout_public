"""일시적 조회 실패를 '구조 없음'으로 굳히지 않는지 확인한다.

PubChem PUG REST 는 초당 5회를 넘기면 503 PUGREST.ServerBusy 를 준다. 그것은
"이 성분은 구조가 없다"가 아니라 "지금 바쁘다"이다. 예전 판본은 둘을 구분하지
않고 모두 None 으로 캐시했고, 그 결과 캐시 50,156 항목 중 49,638개(99.0%)가
영구적으로 "구조 없음"이 되었다. 그중에는 나이아신아마이드·레티놀·코직산·
아데노신처럼 지금 바로 조회되는 성분들이 들어 있었고, 후보 라이브러리가
37,071 종에서 681 종으로 줄었다.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import stage0_cosing as cosing  # noqa: E402


def _response(status: int, payload: object = None):
    return SimpleNamespace(
        status_code=status,
        json=lambda: payload,
        text="busy" if status >= 500 else "",
    )


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cosing.time, "sleep", lambda _s: None)
    monkeypatch.setattr(cosing.time, "monotonic", lambda: 0.0)


def test_a_rate_limit_response_is_not_cached_as_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """503 을 None 으로 캐시하면 그 성분은 다음 실행에서도 시도되지 않는다."""
    monkeypatch.setattr(cosing.requests, "get", lambda *_a, **_k: _response(503))
    cache: dict = {}
    assert cosing._smiles_from_identifier("KOJIC ACID", cache) is None
    assert cache == {}, f"일시적 실패를 캐시했습니다: {cache}"


def test_a_timeout_is_not_cached_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a, **_k):
        raise cosing.requests.RequestException("timed out")

    monkeypatch.setattr(cosing.requests, "get", boom)
    cache: dict = {}
    assert cosing._smiles_from_identifier("ADENOSINE", cache) is None
    assert cache == {}


def test_a_real_404_is_cached_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """진짜 없는 이름은 캐시해도 된다. 그래야 다음 실행이 빨라진다."""
    calls: list[str] = []

    def get(url, **_k):
        calls.append(url)
        return _response(404)

    monkeypatch.setattr(cosing.requests, "get", get)
    cache: dict = {}
    assert cosing._smiles_from_identifier("PICHIA/RETINOL FERMENT EXTRACT", cache) is None
    assert cache == {"smiles_for:PICHIA/RETINOL FERMENT EXTRACT": None}
    assert len(calls) == 1, "404 를 재시도하면 안 됩니다"


def test_a_transient_failure_is_retried_before_giving_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """한 번의 503 으로 포기하면 혼잡한 순간에 성분을 통째로 잃는다."""
    seen: list[int] = []
    payload = {"PropertyTable": {"Properties": [{"CanonicalSMILES": "C1=CC=CC=C1"}]}}

    def get(_url, **_k):
        seen.append(len(seen))
        return _response(503) if len(seen) < 3 else _response(200, payload)

    monkeypatch.setattr(cosing.requests, "get", get)
    cache: dict = {}
    assert cosing._smiles_from_identifier("NIACINAMIDE", cache) == "C1=CC=CC=C1"
    assert len(seen) == 3
    assert cache["smiles_for:NIACINAMIDE"] == "C1=CC=CC=C1"


def test_cid_lookup_also_refuses_to_cache_a_transient_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cosing.requests, "get", lambda *_a, **_k: _response(429))
    cache: dict = {}
    assert cosing._cid_from_name("RETINOL", cache) is None
    assert cache == {}


def test_retry_unresolved_flag_exists_and_keeps_a_backup() -> None:
    """이미 굳어 버린 캐시를 되살릴 수단이 있어야 합니다."""
    source = (ROOT / "scripts/stage0_cosing.py").read_text()
    assert '"--retry-unresolved"' in source
    assert ".before-retry" in source, "이전 캐시를 남기지 않으면 되돌릴 수 없습니다"


def test_the_rate_limiter_holds_under_the_thread_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """수집은 스레드 8개로 돈다. 잠금이 없으면 제한이 새어 다시 503 을 부른다."""
    import threading

    source = (ROOT / "scripts/stage0_cosing.py").read_text()
    assert "_RATE_LOCK" in source, "속도 제한에 잠금이 없습니다"
    assert isinstance(cosing._RATE_LOCK, type(threading.Lock())), "잠금이 아닙니다"
    # 잠금 구간 안에서 마지막 요청 시각을 갱신해야 두 스레드가 같은 값을 읽지 않는다.
    body = source[source.index("def _pubchem_get"):source.index("def _cid_from_name")]
    lock_at = body.index("with _RATE_LOCK:")
    stamp_at = body.index("_LAST_REQUEST_AT[0] = time.monotonic()")
    send_at = body.index("requests.get(url")
    assert lock_at < stamp_at < send_at, (
        "요청을 보내기 전에, 잠금 안에서 시각을 갱신해야 합니다"
    )


def test_the_threaded_path_survives_a_transient_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """수집은 스레드 풀로 돌고, 그 경로는 _fetch_smiles_from_identifier 를 직접 부른다.

    그 함수가 예외를 올리도록 바꿔 놓고 풀 쪽을 고치지 않으면, future.result() 에서
    예외가 다시 터져 수집 전체가 죽는다. 그 시점에는 출력 parquet 이 이미 지워진
    뒤라 라이브러리를 통째로 잃는다 - 실제로 한 번 그렇게 잃었다.
    """
    payload = {"PropertyTable": {"Properties": [{"CanonicalSMILES": "CCO"}]}}

    def get(url, **_k):
        if "GOOD" in url:
            return _response(200, payload)
        if "MISSING" in url:
            return _response(404)
        return _response(503)          # 계속 바쁜 항목

    monkeypatch.setattr(cosing.requests, "get", get)
    cache: dict = {}
    cache_path = tmp_path / "cache.json"
    cosing._resolve_identifier_set(
        identifiers=["GOOD", "MISSING", "BUSY"],
        cache=cache,
        cache_path=cache_path,
        workers=4,
        progress_every=0,
        label="test",
    )
    assert cache.get("smiles_for:GOOD") == "CCO"
    assert cache.get("smiles_for:MISSING") is None
    assert "smiles_for:BUSY" not in cache, "일시적 실패를 캐시했습니다"

    # 함수 이름이 다를 수 있으므로 소스로도 계약을 고정한다.
    source = (ROOT / "scripts/stage0_cosing.py").read_text()
    body = source[source.index("with ThreadPoolExecutor"):source.index("def _ecfp4_words")]
    assert "except TransientLookupError" in body, (
        "스레드 경로가 일시적 실패를 잡지 않으면 수집 전체가 죽습니다"
    )
    assert "continue" in body, "일시적 실패 항목은 캐시하지 않고 넘어가야 합니다"
