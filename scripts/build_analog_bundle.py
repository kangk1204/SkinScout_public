#!/usr/bin/env python3
"""build_analog_bundle.py — 대체소재 검색에 필요한 데이터만 묶는다.

전체 파이프라인의 Stage 0 은 125 GB 를 만든다(ChEMBL 두 판본 64 GB, BindingDB
20 GB, AlphaFold 13 GB, 포켓·PDBQT 17 GB …). 그런데 대체소재 검색 화면이 실제로
여는 것은 세 가지뿐이다:

    data/cosing/cosing.parquet              등재 원료 구조 라이브러리
    data/cosing/admet_cache.parquet         원료별 ADMET 예측 (안전 축)
    data/similarity_index_202609/           활성 측정 인덱스 (근거 축)

앞의 둘은 작고, 셋째는 크지만 **다시 만들 수 없다** - ChEMBL·BindingDB 원본
55 GB 에서 106 만 리간드를 지문화한 결과이기 때문이다. 그래서 받는 쪽이 다시
만들게 하지 않고 만들어진 것을 넘긴다.

`scripts/build_public_snapshot.sh` 와 같은 태도다: 미러가 아니라 스냅샷이고,
받은 쪽이 무결성을 스스로 확인할 수 있어야 한다. 그래서 tar 하나만 내지 않고
파일별 SHA256 과 행 수를 담은 매니페스트를 함께 낸다.

    python scripts/build_analog_bundle.py --out dist/

받는 쪽은 `scripts/fetch_analog_bundle.py` 로 푼다.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("analog.bundle")

BUNDLE_NAME = "skinscout-analog-bundle.tar.gz"
MANIFEST_NAME = "skinscout-analog-bundle.manifest.json"
SCHEMA = "skinscout.analog-bundle.v1"

# 번들에 들어가는 것. 경로는 저장소 루트 기준이고, 푸는 쪽도 같은 자리에 놓는다.
MEMBERS = (
    "data/cosing/cosing.parquet",
    "data/cosing/admet_cache.parquet",
    "data/similarity_index_202609/manifest.json",
    "data/similarity_index_202609/ligands.parquet",
    "data/similarity_index_202609/fingerprints.npy",
)

# ADMET 캐시가 라이브러리를 이만큼도 못 덮으면 묶지 않는다. 덮지 못한 원료는
# 화면에서 안전 축이 통째로 중앙값으로 채워지는데, 그 상태를 "예측이 붙었다"로
# 넘기면 받는 쪽은 재 보지 않은 값을 잰 값으로 읽는다. 키를 맞춘 뒤 실측은 100% 다.
MIN_ADMET_COVERAGE = 0.80


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parquet_rows(path: Path) -> int | None:
    try:
        import pandas as pd

        return int(len(pd.read_parquet(path, columns=["inchikey"])))
    except Exception:                                  # noqa: BLE001
        return None


def admet_coverage() -> tuple[int, int, float]:
    """라이브러리 중 ADMET 예측이 붙는 비율. 묶기 전에 확인한다.

    `cosing.parquet` 의 `inchikey` 열이 아니라 **화면이 쓰는 라이브러리**를 기준으로
    센다. 둘은 다르다 - `load_ingredient_library()` 는 SMILES 를 정규화한 뒤
    InChIKey 를 다시 계산하고 같은 구조를 합치므로 10,120 행이 7,484 종이 된다.
    원본 기준으로 세면 제대로 만든 캐시가 74% 로 보이고, 반대로 원본 키로 만든
    캐시는 100% 로 보인다 - 화면에서는 12.4% 가 비는데도.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    import pandas as pd
    from alternative_ingredients import load_ingredient_library

    library = load_ingredient_library(ROOT / "data/cosing/cosing.parquet")
    cache = pd.read_parquet(ROOT / "data/cosing/admet_cache.parquet", columns=["inchikey"])
    keys = set(cache["inchikey"].astype(str))
    covered = int(library.frame["inchikey"].astype(str).isin(keys).sum())
    total = int(len(library.frame))
    return covered, total, (covered / total if total else 0.0)


def check_inputs() -> list[Path]:
    missing = [name for name in MEMBERS if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit(
            "번들에 넣을 파일이 없습니다:\n  "
            + "\n  ".join(missing)
            + "\n\ncosing.parquet 은 `python scripts/run_skinscout.py --preset stage0`,\n"
            "admet_cache.parquet 은 `python scripts/build_admet_cache.py`,\n"
            "유사도 인덱스는 `python scripts/build_similarity_index.py` 가 만듭니다."
        )
    empty = [name for name in MEMBERS if (ROOT / name).stat().st_size == 0]
    if empty:
        raise SystemExit("번들에 넣을 파일이 비어 있습니다: " + ", ".join(empty))
    return [ROOT / name for name in MEMBERS]


def check_coverage() -> tuple[int, int, float]:
    """묶기 **전에** 부른다. 실패한 빌드가 tar 만 남겨 두면 안 된다."""
    covered, total, fraction = admet_coverage()
    if fraction < MIN_ADMET_COVERAGE:
        raise SystemExit(
            f"ADMET 캐시가 라이브러리의 {fraction:.1%} 만 덮습니다"
            f"(하한 {MIN_ADMET_COVERAGE:.0%}). `python scripts/build_admet_cache.py` 로 "
            "다시 만든 뒤 묶으세요 - 덮이지 않은 원료는 화면에서 안전 축이 통째로 "
            "추정값이 됩니다."
        )
    return covered, total, fraction


def build_manifest(bundle: Path, coverage: tuple[int, int, float]) -> dict:
    covered, total, fraction = coverage
    files = {}
    for name in MEMBERS:
        path = ROOT / name
        record: dict[str, object] = {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        if path.suffix == ".parquet":
            rows = parquet_rows(path)
            if rows is not None:
                record["rows"] = rows
        files[name] = record
    return {
        "schema_version": SCHEMA,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "bundle": {"name": bundle.name, "sha256": sha256(bundle), "bytes": bundle.stat().st_size},
        "files": files,
        "admet_coverage": {"covered": covered, "library": total, "fraction": round(fraction, 4)},
        "note": (
            "대체소재 검색 전용 데이터입니다. 표적 예측·도킹·MD 에 필요한 Stage 0 "
            "산출물은 들어 있지 않습니다."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "dist",
                        help="번들과 매니페스트를 놓을 디렉터리")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    paths = check_inputs()
    coverage = check_coverage()
    LOG.info("ADMET 커버리지 %d/%d (%.1f%%)", coverage[0], coverage[1], 100 * coverage[2])
    args.out.mkdir(parents=True, exist_ok=True)
    bundle = args.out / BUNDLE_NAME
    tmp = bundle.with_suffix(bundle.suffix + ".tmp")
    tmp.unlink(missing_ok=True)

    total_bytes = sum(p.stat().st_size for p in paths)
    LOG.info("묶는 중: %d개 파일 · 원본 %.0f MB", len(paths), total_bytes / 1048576)
    # 압축률은 낮다(지문 259 MB 는 사실상 난수 비트다). gzip 을 쓰는 이유는 크기가
    # 아니라 **받는 쪽에 아무것도 깔지 않아도 되는 것**이다 - 표준 라이브러리로 푼다.
    with tarfile.open(tmp, "w:gz", compresslevel=6) as tar:
        for name in MEMBERS:
            tar.add(ROOT / name, arcname=name)
    tmp.replace(bundle)

    manifest = build_manifest(bundle, coverage)
    (args.out / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    LOG.info("번들 %s (%.0f MB)", bundle, bundle.stat().st_size / 1048576)
    LOG.info("매니페스트 %s", args.out / MANIFEST_NAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
