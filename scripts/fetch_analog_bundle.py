#!/usr/bin/env python3
"""fetch_analog_bundle.py — 대체소재 검색용 데이터 번들을 받아 풀고 확인한다.

`scripts/build_analog_bundle.py` 가 만든 tar.gz 와 매니페스트를 받아, **풀기 전에**
번들 해시를 대조하고, **저장소에 반영하기 전에** 임시 위치에서 파일별 해시를 다시
대조한다. 하나라도 어긋나면 기존 파일은 건드리지 않는다. README 가 설치에 대해
약속한 규칙과 같다 - "받은 파일은 전부 해시로 대조합니다. 중간에 하나라도 어긋나면
그 자리에서 멈춥니다."

    # 로컬 파일에서
    python scripts/fetch_analog_bundle.py --from dist/skinscout-analog-bundle.tar.gz

    # URL 에서 (매니페스트는 같은 자리에서 자동으로 찾는다. 별도 채널에서 받은
    # 매니페스트 해시를 --manifest-sha256 으로 함께 줘야 한다)
    python scripts/fetch_analog_bundle.py --from https://example.org/skinscout-analog-bundle.tar.gz \
        --manifest-sha256 <별도 채널에서 받은 64자리 hex>

    # 이미 깔린 것이 매니페스트와 맞는지만 확인
    python scripts/fetch_analog_bundle.py --check --manifest dist/skinscout-analog-bundle.manifest.json

tar 를 푸는 것은 신뢰 경계다. 보안 경계는 세 겹이다.

1. **매니페스트 신뢰.** 번들에 들어갈 수 있는 파일은 코드에 고정한 `data/` 아래
   다섯 개뿐이다(`EXPECTED_FILES`). 매니페스트가 그 밖을 가리키면 받지 않는다.
   원격 매니페스트는 번들과 같은 곳에서 오므로 그 자체로는 믿을 수 없다 - 별도
   채널에서 받은 SHA256(`--manifest-sha256` 또는
   `SKINSCOUT_ANALOG_MANIFEST_SHA256`)과 일치해야 한다. 신뢰 사슬은
   `별도 digest → 매니페스트 → 번들 해시 → 파일별 해시`다.
2. **멤버 검증.** tar 항목은 일반 파일이어야 하고 `data/` 안의 상대경로여야 한다.
   절대경로, `..`, 심볼릭/하드 링크, 매니페스트에 없는 항목은 거부한다.
3. **원자적 반영.** 저장소 안 임시 디렉터리에 풀어 크기·해시를 전부 확인한 뒤에야
   `os.replace` 로 하나씩 반영한다. 반영 도중 실패하면 백업에서 되돌리므로 기존
   설치가 반쯤 바뀐 채 남지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import logging
import os
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("analog.fetch")

MANIFEST_SUFFIX = ".manifest.json"
SCHEMA = "skinscout.analog-bundle.v1"
MANIFEST_DIGEST_ENV = "SKINSCOUT_ANALOG_MANIFEST_SHA256"
DATA_PREFIX = "data"
SHA256_HEX = set("0123456789abcdef")
MANIFEST_MAX_BYTES = 16 * 1024 * 1024
BUNDLE_TIMEOUT_SECONDS = 600.0

# 코드에 고정한 허용 집합. `scripts/build_analog_bundle.py` 의 MEMBERS 와 같아야
# 한다 - 번들이 문서상 싣는 파일만 설치한다. 매니페스트는 이 집합의 부분집합만
# 가리킬 수 있다(작은 시험 번들이 전체 다섯 개를 담지 않아도 된다).
EXPECTED_FILES = frozenset({
    "data/cosing/cosing.parquet",
    "data/cosing/admet_cache.parquet",
    "data/similarity_index_202609/manifest.json",
    "data/similarity_index_202609/ligands.parquet",
    "data/similarity_index_202609/fingerprints.npy",
})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def _is_valid_sha256(value: object) -> bool:
    text = value if isinstance(value, str) else ""
    lowered = text.lower()
    return len(lowered) == 64 and all(ch in SHA256_HEX for ch in lowered) and len(set(lowered)) > 1


def _under_data(name: object) -> bool:
    """`data/` 안의 상대경로인가. 절대경로와 `..` 는 경계 밖이다."""
    if not isinstance(name, str) or not name or "\x00" in name:
        return False
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        return False
    return path.parts[:1] == (DATA_PREFIX,)


def _trusted_digest(cli_value: str | None) -> str | None:
    value = (cli_value or os.environ.get(MANIFEST_DIGEST_ENV, "")).strip().lower()
    if not value:
        return None
    if not _is_valid_sha256(value):
        raise SystemExit("--manifest-sha256 는 64자리 16진수여야 합니다")
    return value


def _read_chunk(source, size: int) -> bytes:
    if isinstance(source, http.client.HTTPResponse):
        return source.read1(size)
    return source.read(size)


def _copy_limited(
    source,
    destination,
    *,
    expected_bytes: int,
    deadline: float,
    label: str,
) -> None:
    """Copy at most ``expected_bytes``; never follow the stream to EOF first."""

    written = 0
    while True:
        if time.monotonic() > deadline:
            raise SystemExit(f"내려받기가 제한 시간을 넘었습니다: {label}")
        remaining = expected_bytes - written
        chunk = _read_chunk(source, min(1024 * 1024, max(remaining, 0) + 1))
        if not chunk:
            return
        written += len(chunk)
        if written > expected_bytes:
            raise SystemExit(
                f"내려받은 크기가 매니페스트의 {expected_bytes} bytes를 넘었습니다: {label}"
            )
        destination.write(chunk)


def _read_limited(response, *, limit: int, label: str) -> bytes:
    data = response.read(limit + 1)
    if len(data) > limit:
        raise SystemExit(f"{label}가 너무 큽니다 (>{limit} bytes)")
    return data


def _retrieve(
    source: str,
    destination: Path,
    *,
    expected_bytes: int,
    timeout: float = BUNDLE_TIMEOUT_SECONDS,
) -> Path:
    if _is_url(source):
        LOG.info("내려받는 중: %s", source)
        deadline = time.monotonic() + timeout
        try:
            with urllib.request.urlopen(source, timeout=timeout) as response:
                with destination.open("wb") as out:
                    _copy_limited(
                        response,
                        out,
                        expected_bytes=expected_bytes,
                        deadline=deadline,
                        label="bundle",
                    )
        except TimeoutError as exc:
            destination.unlink(missing_ok=True)
            raise SystemExit(f"내려받기가 제한 시간을 넘었습니다: {source}") from exc
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
        return destination
    path = Path(source).expanduser()
    if not path.is_file():
        raise SystemExit(f"번들 파일이 없습니다: {path}")
    return path


def _validate_manifest(manifest: dict) -> None:
    if manifest.get("schema_version") != SCHEMA:
        raise SystemExit(
            f"매니페스트 형식이 다릅니다: {manifest.get('schema_version')!r} (기대 {SCHEMA!r})"
        )
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise SystemExit("매니페스트에 files 가 없습니다")
    for name, record in files.items():
        if name not in EXPECTED_FILES or not _under_data(name):
            raise SystemExit(
                f"매니페스트가 허용 목록 밖의 파일을 가리킵니다: {name!r}\n"
                "  받는 쪽은 data/ 아래 정해진 파일만 설치합니다."
            )
        if not isinstance(record, dict):
            raise SystemExit(f"매니페스트 항목이 객체가 아닙니다: {name!r}")
        if not _is_valid_sha256(record.get("sha256")):
            raise SystemExit(f"매니페스트 항목의 sha256 이 올바르지 않습니다: {name!r}")
        size = record.get("bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise SystemExit(f"매니페스트 항목의 bytes 가 올바르지 않습니다: {name!r}")
    bundle = manifest.get("bundle")
    if not isinstance(bundle, dict) or not _is_valid_sha256(bundle.get("sha256")):
        raise SystemExit("매니페스트에 번들 해시가 없습니다")
    bundle_bytes = bundle.get("bytes")
    if not isinstance(bundle_bytes, int) or isinstance(bundle_bytes, bool) or bundle_bytes < 0:
        raise SystemExit("매니페스트에 번들 크기(bytes)가 없습니다")


def _load_manifest(source: str | None, bundle_source: str, trusted_digest: str | None) -> dict:
    if source is None:
        # 번들 옆에 같은 이름으로 있다고 본다. 없으면 사용자가 직접 줘야 한다.
        if _is_url(bundle_source):
            base = bundle_source.rsplit(".tar.gz", 1)[0]
            source = base + MANIFEST_SUFFIX
        else:
            source = str(Path(bundle_source).with_suffix("").with_suffix("")) + MANIFEST_SUFFIX
    if _is_url(source):
        LOG.info("매니페스트: %s", source)
        with urllib.request.urlopen(source, timeout=120) as response:
            raw = _read_limited(response, limit=MANIFEST_MAX_BYTES, label="매니페스트")
    else:
        path = Path(source).expanduser()
        if not path.is_file():
            raise SystemExit(
                f"매니페스트가 없습니다: {path}\n"
                "번들만으로는 무결성을 확인할 수 없습니다. --manifest 로 지정하세요."
            )
        raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if trusted_digest is not None:
        if digest != trusted_digest:
            raise SystemExit(
                "매니페스트 해시가 신뢰된 값과 다릅니다. 받지 않았습니다.\n"
                f"  받은 것  : {digest}\n  신뢰된 값: {trusted_digest}"
            )
        LOG.info("매니페스트가 신뢰된 digest 와 일치합니다")
    elif _is_url(source):
        raise SystemExit(
            "원격 매니페스트는 번들과 같은 곳에서 오므로 그대로 믿을 수 없습니다.\n"
            "별도 채널에서 받은 SHA256 을 --manifest-sha256 (또는 "
            f"{MANIFEST_DIGEST_ENV}) 로 지정하세요."
        )
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"매니페스트를 읽을 수 없습니다: {exc}")
    if not isinstance(manifest, dict):
        raise SystemExit("매니페스트는 JSON 객체여야 합니다")
    _validate_manifest(manifest)
    return manifest


def _safe_members(tar: tarfile.TarFile, expected: set[str]) -> list[tarfile.TarInfo]:
    """매니페스트에 적힌 일반 파일만, 경로가 안전할 때만 통과시킨다."""
    members = []
    seen: set[str] = set()
    for member in tar.getmembers():
        name = member.name
        if name not in expected:
            raise SystemExit(f"번들에 매니페스트가 모르는 항목이 있습니다: {name!r}")
        if name in seen:
            raise SystemExit(f"번들에 같은 항목이 두 번 있습니다: {name!r}")
        if not _under_data(name) or name not in EXPECTED_FILES:
            raise SystemExit(f"번들 항목이 허용 범위 밖입니다: {name!r}")
        if member.issym() or member.islnk():
            raise SystemExit(f"번들에 링크 항목이 있습니다: {name!r}")
        if not member.isfile():
            raise SystemExit(f"번들에 일반 파일이 아닌 항목이 있습니다: {name!r}")
        members.append(member)
        seen.add(name)
    missing = expected - seen
    if missing:
        raise SystemExit("번들에 빠진 항목이 있습니다: " + ", ".join(sorted(missing)))
    return members


def _extract_and_verify(
    tar: tarfile.TarFile,
    members: list[tarfile.TarInfo],
    staging: Path,
    records: dict,
) -> None:
    """임시 위치에 풀고 크기·해시를 확인한다. 저장소는 아직 건드리지 않는다."""
    for member in members:
        source = tar.extractfile(member)
        if source is None:
            raise SystemExit(f"번들 항목을 읽을 수 없습니다: {member.name!r}")
        target = staging / member.name
        target.parent.mkdir(parents=True, exist_ok=True)
        with source, target.open("wb") as out:
            shutil.copyfileobj(source, out, length=1024 * 1024)
    problems = []
    for name, record in records.items():
        path = staging / name
        if path.is_symlink() or not path.is_file():
            problems.append(f"{name}: 풀린 파일이 없음")
        elif path.stat().st_size != record["bytes"]:
            problems.append(f"{name}: 크기가 다름")
        elif sha256(path) != record["sha256"]:
            problems.append(f"{name}: 해시가 다름")
    if problems:
        raise SystemExit(
            "푼 뒤 확인에서 어긋났습니다. 저장소에는 아무것도 반영하지 않았습니다:\n  "
            + "\n  ".join(problems)
        )


def verify_installed(manifest: dict) -> list[str]:
    """깔려 있는 파일이 매니페스트와 맞는지. 어긋난 것의 목록을 돌려준다."""
    problems = []
    for name, record in manifest["files"].items():
        path = ROOT / name
        if path.is_symlink():
            problems.append(f"{name}: 심볼릭 링크")
            continue
        if not path.is_file():
            problems.append(f"{name}: 없음")
            continue
        if path.stat().st_size != record["bytes"]:
            problems.append(f"{name}: 크기가 다름")
            continue
        if sha256(path) != record["sha256"]:
            problems.append(f"{name}: 해시가 다름")
    return problems


def _prepare_target_parent(target: Path) -> None:
    """부모 디렉터리에 심볼릭 링크가 끼어 저장소 밖으로 새는 것을 막는다."""
    try:
        relative = target.relative_to(ROOT)
    except ValueError:
        raise SystemExit(f"설치 경로가 저장소 밖을 가리킵니다: {target}")
    current = ROOT
    for part in relative.parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise SystemExit(f"설치 경로에 심볼릭 링크가 있습니다: {current}")
        if current.exists() and not current.is_dir():
            raise SystemExit(f"설치 경로의 부모가 디렉터리가 아닙니다: {current}")
        current.mkdir(exist_ok=True)


def _restore(applied: list[Path], backups: list[tuple[Path, Path]]) -> None:
    for target in reversed(applied):
        try:
            if target.is_symlink() or target.exists():
                target.unlink()
        except OSError as exc:
            LOG.error("되돌리지 못했습니다: %s (%s)", target, exc)
    for target, backup in reversed(backups):
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(backup, target)
        except OSError as exc:
            LOG.error("백업을 되돌리지 못했습니다: %s (%s)", target, exc)


def _apply_staged(staging: Path, records: dict) -> tuple[list[Path], list[tuple[Path, Path]]]:
    """검증이 끝난 파일만 저장소로 옮긴다. 실패하면 그 자리에서 되돌린다."""
    backup_root = staging / "__rollback__"
    applied: list[Path] = []
    backups: list[tuple[Path, Path]] = []
    try:
        for name in sorted(records):
            source = staging / name
            target = ROOT / name
            _prepare_target_parent(target)
            if target.is_symlink() or target.exists():
                if target.is_dir() and not target.is_symlink():
                    raise SystemExit(f"설치 대상이 디렉터리입니다: {target}")
                backup = backup_root / name
                backup.parent.mkdir(parents=True, exist_ok=True)
                os.replace(target, backup)
                backups.append((target, backup))
            os.replace(source, target)
            applied.append(target)
        return applied, backups
    except BaseException:
        _restore(applied, backups)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="source", help="번들 tar.gz 의 URL 또는 로컬 경로")
    parser.add_argument("--manifest", default=None, help="매니페스트 URL 또는 경로")
    parser.add_argument("--manifest-sha256", default=None,
                        help="별도 채널에서 받은 매니페스트 SHA256 (원격 매니페스트에는 필수)")
    parser.add_argument("--check", action="store_true",
                        help="내려받지 않고, 이미 깔린 것이 매니페스트와 맞는지만 확인")
    parser.add_argument("--force", action="store_true",
                        help="이미 있는 파일을 덮어쓴다")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    trusted_digest = _trusted_digest(args.manifest_sha256)

    if args.check:
        if args.manifest is None:
            raise SystemExit("--check 에는 --manifest 가 필요합니다")
        manifest = _load_manifest(args.manifest, "", trusted_digest)
        problems = verify_installed(manifest)
        if problems:
            LOG.error("맞지 않는 항목 %d개:\n  %s", len(problems), "\n  ".join(problems))
            return 1
        LOG.info("확인됨: %d개 파일이 매니페스트와 일치합니다", len(manifest["files"]))
        return 0

    if not args.source:
        raise SystemExit("--from 으로 번들 위치를 주세요(또는 --check 를 쓰세요)")
    manifest = _load_manifest(args.manifest, args.source, trusted_digest)

    existing = [name for name in manifest["files"] if (ROOT / name).is_file()]
    if existing and not args.force:
        raise SystemExit(
            "이미 있는 파일을 덮어쓰지 않습니다:\n  " + "\n  ".join(existing)
            + "\n\n덮어쓰려면 --force, 지금 것이 맞는지만 보려면 --check 를 쓰세요."
        )

    # 임시 디렉터리를 저장소 안에 만든다 - os.replace 가 다른 파일시스템을 넘지
    # 않아야 반영이 원자적이다.
    with tempfile.TemporaryDirectory(prefix=".skinscout-bundle-", dir=str(ROOT)) as staging_text:
        staging = Path(staging_text)
        staged = _retrieve(
            args.source,
            staging / "bundle.tar.gz",
            expected_bytes=int(manifest["bundle"]["bytes"]),
        )
        digest = sha256(staged)
        if digest != manifest["bundle"]["sha256"]:
            raise SystemExit(
                "번들 해시가 매니페스트와 다릅니다. 풀지 않았습니다.\n"
                f"  받은 것 : {digest}\n  매니페스트: {manifest['bundle']['sha256']}"
            )
        LOG.info("번들 해시 확인됨 (%.0f MB)", staged.stat().st_size / 1048576)
        records = manifest["files"]
        with tarfile.open(staged, "r:gz") as tar:
            members = _safe_members(tar, set(records))
            _extract_and_verify(tar, members, staging, records)
        LOG.info("파일 %d개를 임시 위치에서 확인했습니다", len(records))
        applied, backups = _apply_staged(staging, records)
        problems = verify_installed(manifest)
        if problems:
            _restore(applied, backups)
            raise SystemExit("반영 뒤 확인에서 어긋나 되돌렸습니다:\n  " + "\n  ".join(problems))

    coverage = manifest.get("admet_coverage", {})
    LOG.info("%d개 파일을 풀고 확인했습니다", len(manifest["files"]))
    if coverage:
        LOG.info("ADMET 커버리지 %s/%s (%.1f%%)",
                 coverage.get("covered"), coverage.get("library"),
                 100 * float(coverage.get("fraction", 0.0)))
    LOG.info("이제 `python3 scripts/run_workbench.py` 로 대체소재 검색을 쓸 수 있습니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
