#!/usr/bin/env python3
"""fetch_analog_bundle.py — 대체소재 검색용 데이터 번들을 받아 풀고 확인한다.

`scripts/build_analog_bundle.py` 가 만든 tar.gz 와 매니페스트를 받아, **풀기 전에**
번들 해시를 대조하고, 푼 뒤에 파일별 해시를 다시 대조한다. 하나라도 어긋나면
남기지 않는다. README 가 설치에 대해 약속한 규칙과 같다 - "받은 파일은 전부
해시로 대조합니다. 중간에 하나라도 어긋나면 그 자리에서 멈춥니다."

    # 로컬 파일에서
    python scripts/fetch_analog_bundle.py --from dist/skinscout-analog-bundle.tar.gz

    # URL 에서 (매니페스트는 같은 자리에서 자동으로 찾는다)
    python scripts/fetch_analog_bundle.py --from https://example.org/skinscout-analog-bundle.tar.gz

    # 이미 깔린 것이 매니페스트와 맞는지만 확인
    python scripts/fetch_analog_bundle.py --check --manifest dist/skinscout-analog-bundle.manifest.json

tar 를 푸는 것은 신뢰 경계다. 멤버 경로를 하나하나 확인해 `data/` 밖으로 나가는
것, 절대경로, 심볼릭/하드 링크를 전부 거부한다 - 매니페스트에 적힌 다섯 파일만
푼다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("analog.fetch")

MANIFEST_SUFFIX = ".manifest.json"
SCHEMA = "skinscout.analog-bundle.v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def _retrieve(source: str, destination: Path) -> Path:
    if _is_url(source):
        LOG.info("내려받는 중: %s", source)
        with urllib.request.urlopen(source, timeout=600) as response, destination.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
        return destination
    path = Path(source).expanduser()
    if not path.is_file():
        raise SystemExit(f"번들 파일이 없습니다: {path}")
    return path


def _load_manifest(source: str | None, bundle_source: str) -> dict:
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
            manifest = json.loads(response.read().decode("utf-8"))
    else:
        path = Path(source).expanduser()
        if not path.is_file():
            raise SystemExit(
                f"매니페스트가 없습니다: {path}\n"
                "번들만으로는 무결성을 확인할 수 없습니다. --manifest 로 지정하세요."
            )
        manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA:
        raise SystemExit(
            f"매니페스트 형식이 다릅니다: {manifest.get('schema_version')!r} (기대 {SCHEMA!r})"
        )
    return manifest


def _safe_members(tar: tarfile.TarFile, expected: set[str]) -> list[tarfile.TarInfo]:
    """매니페스트에 적힌 일반 파일만, 경로가 안전할 때만 통과시킨다."""
    members = []
    for member in tar.getmembers():
        name = member.name
        if name not in expected:
            raise SystemExit(f"번들에 매니페스트가 모르는 항목이 있습니다: {name!r}")
        if not member.isfile():
            raise SystemExit(f"번들에 일반 파일이 아닌 항목이 있습니다: {name!r}")
        target = (ROOT / name).resolve()
        if not str(target).startswith(str(ROOT.resolve()) + "/"):
            raise SystemExit(f"번들 항목이 저장소 밖을 가리킵니다: {name!r}")
        members.append(member)
    missing = expected - {m.name for m in members}
    if missing:
        raise SystemExit("번들에 빠진 항목이 있습니다: " + ", ".join(sorted(missing)))
    return members


def verify_installed(manifest: dict) -> list[str]:
    """깔려 있는 파일이 매니페스트와 맞는지. 어긋난 것의 목록을 돌려준다."""
    problems = []
    for name, record in manifest["files"].items():
        path = ROOT / name
        if not path.is_file():
            problems.append(f"{name}: 없음")
            continue
        if path.stat().st_size != record["bytes"]:
            problems.append(f"{name}: 크기가 다름")
            continue
        if sha256(path) != record["sha256"]:
            problems.append(f"{name}: 해시가 다름")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="source", help="번들 tar.gz 의 URL 또는 로컬 경로")
    parser.add_argument("--manifest", default=None, help="매니페스트 URL 또는 경로")
    parser.add_argument("--check", action="store_true",
                        help="내려받지 않고, 이미 깔린 것이 매니페스트와 맞는지만 확인")
    parser.add_argument("--force", action="store_true",
                        help="이미 있는 파일을 덮어쓴다")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.check:
        if args.manifest is None:
            raise SystemExit("--check 에는 --manifest 가 필요합니다")
        manifest = _load_manifest(args.manifest, "")
        problems = verify_installed(manifest)
        if problems:
            LOG.error("맞지 않는 항목 %d개:\n  %s", len(problems), "\n  ".join(problems))
            return 1
        LOG.info("확인됨: %d개 파일이 매니페스트와 일치합니다", len(manifest["files"]))
        return 0

    if not args.source:
        raise SystemExit("--from 으로 번들 위치를 주세요(또는 --check 를 쓰세요)")
    manifest = _load_manifest(args.manifest, args.source)

    existing = [name for name in manifest["files"] if (ROOT / name).is_file()]
    if existing and not args.force:
        raise SystemExit(
            "이미 있는 파일을 덮어쓰지 않습니다:\n  " + "\n  ".join(existing)
            + "\n\n덮어쓰려면 --force, 지금 것이 맞는지만 보려면 --check 를 쓰세요."
        )

    with tempfile.TemporaryDirectory(prefix="skinscout-bundle-") as tmpdir:
        staged = _retrieve(args.source, Path(tmpdir) / "bundle.tar.gz")
        digest = sha256(staged)
        if digest != manifest["bundle"]["sha256"]:
            raise SystemExit(
                "번들 해시가 매니페스트와 다릅니다. 풀지 않았습니다.\n"
                f"  받은 것 : {digest}\n  매니페스트: {manifest['bundle']['sha256']}"
            )
        LOG.info("번들 해시 확인됨 (%.0f MB)", staged.stat().st_size / 1048576)
        with tarfile.open(staged, "r:gz") as tar:
            members = _safe_members(tar, set(manifest["files"]))
            tar.extractall(ROOT, members=members)

    problems = verify_installed(manifest)
    if problems:
        raise SystemExit("푼 뒤 확인에서 어긋났습니다:\n  " + "\n  ".join(problems))
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
