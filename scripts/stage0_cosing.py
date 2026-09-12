#!/usr/bin/env python3
"""stage0_cosing.py — Ingest CosIng INCI database → ECFP4-indexed parquet.

Pipeline:
  1. Load <OUTDIR>/cosing.csv (operator-supplied — fetched from EU data portal).
  2. Resolve CAS/INCI → canonical SMILES via PubChem PUG REST (with disk cache).
  3. Compute ECFP4 + Bemis-Murcko scaffold for each SMILES.
  4. Write <OUTDIR>/cosing.parquet with columns:
       inci_name, cas, einecs, functions, smiles, inchikey,
       ecfp4 (32 decimal uint64 words), scaffold_smiles
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

LOG = logging.getLogger("stage0.cosing")
PUG_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _remove_outputs(*paths: Path) -> None:
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)


def _write_parquet_atomic(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


# PubChem PUG REST 는 초당 5회를 넘기면 503 PUGREST.ServerBusy 를 준다. 그 503 은
# "이 성분은 구조가 없다"가 아니라 "지금 바쁘다"이다. 둘을 구분하지 않으면 잠깐의
# 혼잡이 영구적인 "구조 없음"으로 굳는다 - 실측으로 캐시 50,156 항목 중 49,638개
# (99.0%)가 그렇게 굳었고, 그중에는 나이아신아마이드·레티놀·코직산·아데노신처럼
# PubChem 에서 지금 바로 조회되는 성분들이 들어 있다.
PUBCHEM_MIN_INTERVAL_SECONDS = 0.25   # 초당 4회
PUBCHEM_MAX_ATTEMPTS = 4
_LAST_REQUEST_AT = [0.0]
# 수집은 스레드 8개로 돈다. 잠금이 없으면 여러 스레드가 같은 시각을 읽고 동시에
# 나가서 제한이 새고, 그러면 다시 503 을 부른다 - 고치려던 문제로 되돌아간다.
_RATE_LOCK = threading.Lock()


class TransientLookupError(RuntimeError):
    """일시적 실패. 이 결과는 캐시하면 안 된다."""


def _pubchem_get(url: str) -> "requests.Response | None":
    """PubChem 조회. 진짜 없음(404)이면 None, 일시적 실패면 예외를 올린다.

    호출부가 둘을 구분할 수 있어야 한다. 예전에는 모든 실패가 같은 None 이라
    호출부가 그것을 "없음"으로 캐시했다.
    """
    last = ""
    for attempt in range(1, PUBCHEM_MAX_ATTEMPTS + 1):
        with _RATE_LOCK:
            wait = PUBCHEM_MIN_INTERVAL_SECONDS - (time.monotonic() - _LAST_REQUEST_AT[0])
            if wait > 0:
                time.sleep(wait)
            _LAST_REQUEST_AT[0] = time.monotonic()
        try:
            response = requests.get(url, timeout=30)
        except requests.RequestException as exc:
            last = f"{type(exc).__name__}: {exc}"
        else:
            if response.status_code == 200:
                return response
            if response.status_code == 404:
                return None          # PubChem 이 모르는 이름. 이것만 캐시해도 된다.
            last = f"HTTP {response.status_code}"
        if attempt < PUBCHEM_MAX_ATTEMPTS:
            time.sleep(min(2 ** attempt, 8))
    raise TransientLookupError(f"{url} 조회 실패({PUBCHEM_MAX_ATTEMPTS}회): {last}")


def _cid_from_name(name: str, cache: dict) -> int | None:
    if name in cache:
        return cache[name]
    try:
        response = _pubchem_get(
            f"{PUG_BASE}/compound/name/{requests.utils.quote(name)}/cids/JSON"
        )
    except TransientLookupError as exc:
        LOG.warning("CID 조회를 일시적 실패로 건너뜁니다(캐시하지 않음) %s: %s", name, exc)
        return None
    if response is not None:
        try:
            cid = response.json()["IdentifierList"]["CID"][0]
        except (ValueError, KeyError, IndexError):
            cid = None
        if cid is not None:
            cache[name] = cid
            return cid
    cache[name] = None
    return None


def _smiles_from_cid(cid: int, cache: dict) -> str | None:
    if cid in cache:
        return cache[cid]
    try:
        response = _pubchem_get(
            f"{PUG_BASE}/compound/cid/{cid}/property/CanonicalSMILES/JSON"
        )
    except TransientLookupError as exc:
        LOG.warning("SMILES 조회를 일시적 실패로 건너뜁니다(캐시하지 않음) CID %s: %s", cid, exc)
        return None
    if response is not None:
        try:
            smi = response.json()["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
        except (ValueError, KeyError, IndexError):
            smi = None
        if smi:
            cache[cid] = smi
            return smi
    cache[cid] = None
    return None


def _fetch_smiles_from_identifier(identifier: str) -> str | None:
    """이름으로 SMILES 조회. 일시적 실패는 TransientLookupError 로 올린다."""
    response = _pubchem_get(
        f"{PUG_BASE}/compound/name/"
        f"{requests.utils.quote(identifier)}/property/"
        "CanonicalSMILES,IsomericSMILES,InChIKey/JSON"
    )
    if response is None:
        return None
    try:
        props = response.json()["PropertyTable"]["Properties"][0]
    except (ValueError, KeyError, IndexError):
        return None
    smi = (
        props.get("CanonicalSMILES")
        or props.get("SMILES")
        or props.get("ConnectivitySMILES")
        or props.get("IsomericSMILES")
    )
    return str(smi) if smi else None


def _smiles_from_identifier(identifier: str, cache: dict) -> str | None:
    cache_key = f"smiles_for:{identifier}"
    if cache_key in cache:
        return cache[cache_key]
    try:
        resolved = _fetch_smiles_from_identifier(identifier)
    except TransientLookupError as exc:
        # 캐시하지 않는다. 여기서 None 을 굳히면 다음 실행이 다시 시도하지 않고,
        # 그 성분은 영원히 "구조 없음"이 된다.
        LOG.warning("SMILES 조회를 일시적 실패로 건너뜁니다(캐시하지 않음) %s: %s",
                    identifier, exc)
        return None
    cache[cache_key] = resolved
    return resolved


def _identifier_candidates(row: CosingRow) -> list[str]:
    candidates: list[str] = []
    if row.cas:
        candidates.extend(part.strip() for part in row.cas.split(";"))
    candidates.append(row.inci_name)
    seen: dict[str, None] = {}
    for candidate in candidates:
        if candidate and candidate != "-":
            seen.setdefault(candidate, None)
    return list(seen)


def _smiles_from_cache(identifier: str, cache: dict) -> str | None:
    smi = cache.get(f"smiles_for:{identifier}")
    return str(smi).strip() if smi else None


def _row_smiles_from_cache(row: CosingRow, cache: dict) -> str | None:
    for identifier in _identifier_candidates(row):
        smi = _smiles_from_cache(identifier, cache)
        if smi:
            return smi
    return None


def _resolve_identifier_set(
    *,
    identifiers: list[str],
    cache: dict,
    cache_path: Path,
    workers: int,
    progress_every: int,
    label: str,
) -> None:
    pending = [
        identifier
        for identifier in dict.fromkeys(identifiers)
        if f"smiles_for:{identifier}" not in cache
    ]
    if not pending:
        return
    LOG.info("Resolving %d PubChem %s identifiers", len(pending), label)
    done = 0
    transient = 0
    if workers <= 1:
        for identifier in pending:
            _smiles_from_identifier(identifier, cache)
            done += 1
            if progress_every and (done == len(pending) or done % progress_every == 0):
                _write_json_atomic(cache_path, dict(cache))
                LOG.info("PubChem %s progress: %d/%d", label, done, len(pending))
        return

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_fetch_smiles_from_identifier, identifier): identifier
            for identifier in pending
        }
        for future in as_completed(futures):
            identifier = futures[future]
            try:
                resolved = future.result()
            except TransientLookupError as exc:
                # 캐시하지 않는다. 여기서 굳히면 다음 실행이 다시 시도하지 않는다.
                # 예외를 그냥 올리면 수집 전체가 죽는데, 출력 parquet 은 이미
                # 지워진 뒤라 라이브러리를 통째로 잃는다.
                LOG.warning("일시적 실패로 건너뜁니다(캐시하지 않음) %s: %s",
                            identifier, exc)
                transient += 1
                done += 1
                continue
            cache[f"smiles_for:{identifier}"] = resolved
            done += 1
            if progress_every and (done == len(pending) or done % progress_every == 0):
                _write_json_atomic(cache_path, dict(cache))
                LOG.info("PubChem %s progress: %d/%d", label, done, len(pending))
    if transient:
        LOG.warning(
            "%s: %d개는 일시적 실패로 해석하지 못했습니다. 캐시하지 않았으니 "
            "다시 실행하면 그 항목만 재시도합니다", label, transient,
        )


def _ecfp4_words(mol: Chem.Mol) -> list[str]:
    bv = _MORGAN_GENERATOR.GetFingerprint(mol)
    arr = np.zeros(2048, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    packed = np.packbits(arr).view(np.uint64).copy()
    if packed.dtype.byteorder == ">":
        packed = packed.byteswap()
    return [str(int(word)) for word in packed]


@dataclass
class CosingRow:
    inci_name: str
    cas: str | None
    einecs: str | None
    functions: list[str]
    smiles: str | None


def parse_input(in_csv: Path) -> list[CosingRow]:
    if not in_csv.exists():
        return []
    with in_csv.open(newline="", encoding="utf-8") as handle:
        sample = handle.read(8192)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        fieldnames = reader.fieldnames or []
        records = list(reader)
    cols = {c.lower(): c for c in fieldnames if c is not None}
    iname = cols.get("inci name") or cols.get("inci_name") or cols.get("name")
    if iname is None:
        raise ValueError(
            "CosIng CSV must include an INCI name column "
            "('inci name', 'inci_name', or 'name')"
        )
    cas_c = cols.get("cas")
    ein_c = cols.get("einecs")
    fn_c  = cols.get("function") or cols.get("functions")
    rows: list[CosingRow] = []
    blank_name_rows: list[int] = []
    for row_idx, r in enumerate(records):
        raw_name = r.get(iname, "")
        inci_name = str(raw_name or "").strip()
        if not inci_name:
            blank_name_rows.append(row_idx)
        functions = []
        if fn_c:
            functions = [f.strip() for f in str(r.get(fn_c) or "").split(";") if f.strip()]
        cas = str(r.get(cas_c) or "").strip() or None if cas_c else None
        einecs = str(r.get(ein_c) or "").strip() or None if ein_c else None
        rows.append(CosingRow(
            inci_name=inci_name,
            cas=cas,
            einecs=einecs,
            functions=functions,
            smiles=None,
        ))
    if blank_name_rows:
        preview = ", ".join(str(idx) for idx in blank_name_rows[:5])
        if len(blank_name_rows) > 5:
            preview += ", ..."
        raise ValueError(
            "CosIng CSV column 'inci_name' contains blank values at row index "
            f"{preview}"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true",
                        help="Emit a small placeholder DB without hitting PubChem.")
    parser.add_argument("--min-resolution-fraction", type=float, default=0.01)
    parser.add_argument("--min-resolved-count", type=int, default=100)
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--pubchem-workers", type=int, default=8)
    parser.add_argument(
        "--retry-unresolved", action="store_true",
        help="캐시에서 '구조 없음'으로 굳은 항목을 지우고 다시 조회합니다. "
             "일시적 실패(초당 요청 제한, 5xx, 타임아웃)를 영구 결과로 캐시하던 "
             "판본이 만든 캐시를 되살릴 때 씁니다.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    in_csv = args.out_dir / "cosing.csv"
    cache_path = args.out_dir / ".pubchem_cache.json"
    if args.retry_unresolved and cache_path.exists():
        try:
            stale = json.loads(cache_path.read_text())
        except (OSError, ValueError) as exc:
            raise SystemExit(f"캐시를 읽지 못했습니다: {cache_path}: {exc}")
        dropped = {k: v for k, v in stale.items() if v is not None}
        removed = len(stale) - len(dropped)
        backup = cache_path.with_suffix(".json.before-retry")
        backup.write_text(json.dumps(stale))
        _write_json_atomic(cache_path, dropped)
        LOG.info(
            "캐시의 '구조 없음' 항목 %d개를 지우고 다시 조회합니다(%d개는 유지). "
            "이전 캐시는 %s 에 두었습니다",
            removed, len(dropped), backup.name,
        )
    out_parquet = args.out_dir / "cosing.parquet"
    manifest_path = args.out_dir / "cosing_ingest_manifest.json"
    _remove_outputs(out_parquet, manifest_path)
    if not 0.0 <= args.min_resolution_fraction <= 1.0:
        raise SystemExit(
            "--min-resolution-fraction must be in [0, 1]: "
            f"{args.min_resolution_fraction:g}"
        )
    if args.min_resolved_count < 1:
        raise SystemExit(
            f"--min-resolved-count must be positive: {args.min_resolved_count}"
        )
    if args.progress_every < 0:
        raise SystemExit(f"--progress-every must be non-negative: {args.progress_every}")
    if args.pubchem_workers < 1:
        raise SystemExit(f"--pubchem-workers must be positive: {args.pubchem_workers}")
    if args.dry_run:
        LOG.warning("Dry-run requested; emit minimal cosing.parquet stub")
        # Three canonical placeholders to keep downstream code paths alive.
        placeholders = [
            ("Niacinamide", "98-92-0", "NC(=O)c1cccnc1"),
            ("Retinol",     "68-26-8", "CC1=C(/C=C/C(C)=C/C=C/C(C)=C/CO)C(C)(C)CCC1"),
            ("Glycerin",    "56-81-5", "OCC(O)CO"),
        ]
        records = []
        for inci, cas, smi in placeholders:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            records.append({
                "inci_name": inci,
                "cas": cas,
                "einecs": None,
                "functions": "Skin Conditioning",
                "smiles": smi,
                "inchikey": Chem.MolToInchiKey(mol),
                "ecfp4": _ecfp4_words(mol),
                "scaffold_smiles": Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol)),
            })
        _write_parquet_atomic(out_parquet, pd.DataFrame(records))
        _write_json_atomic(
            manifest_path,
            {
                "dry_run": True,
                "input_rows": len(records),
                "resolved_rows": len(records),
                "resolution_fraction": 1.0,
                "unresolved_rows": [],
                "invalid_smiles_rows": [],
            },
        )
        return

    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    rows = parse_input(in_csv)
    if not rows:
        raise SystemExit(
            f"CosIng CSV is required for full ingest and must contain INCI rows: {in_csv}. "
            "Use --dry-run only for explicit placeholder diagnostics."
        )

    cas_identifiers = [
        identifier
        for row in rows
        for identifier in _identifier_candidates(
            CosingRow(
                inci_name="",
                cas=row.cas,
                einecs=None,
                functions=[],
                smiles=None,
            )
        )
    ]
    _resolve_identifier_set(
        identifiers=cas_identifiers,
        cache=cache,
        cache_path=cache_path,
        workers=args.pubchem_workers,
        progress_every=args.progress_every,
        label="CAS",
    )
    name_identifiers = [
        row.inci_name
        for row in rows
        if row.inci_name and _row_smiles_from_cache(row, cache) is None
    ]
    _resolve_identifier_set(
        identifiers=name_identifiers,
        cache=cache,
        cache_path=cache_path,
        workers=args.pubchem_workers,
        progress_every=args.progress_every,
        label="INCI",
    )
    _write_json_atomic(cache_path, dict(cache))

    out_records: list[dict] = []
    unresolved_rows: list[dict[str, object]] = []
    invalid_smiles_rows: list[dict[str, object]] = []
    for i, row in enumerate(rows):
        smi = _row_smiles_from_cache(row, cache)
        if smi is None or not str(smi).strip():
            unresolved_rows.append({"row_index": i, "inci_name": row.inci_name})
            continue
        smi = str(smi).strip()
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid_smiles_rows.append(
                {"row_index": i, "inci_name": row.inci_name, "smiles": smi}
            )
            continue
        out_records.append({
            "inci_name": row.inci_name,
            "cas": row.cas,
            "einecs": row.einecs,
            "functions": ";".join(row.functions),
            "smiles": smi,
            "inchikey": Chem.MolToInchiKey(mol),
            "ecfp4": _ecfp4_words(mol),
            "scaffold_smiles": Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol)),
        })
        done = i + 1
        if args.progress_every and (done == len(rows) or done % args.progress_every == 0):
            LOG.info(
                "CosIng materialization progress: %d/%d resolved=%d "
                "no_smiles=%d invalid=%d",
                done,
                len(rows),
                len(out_records),
                len(unresolved_rows),
                len(invalid_smiles_rows),
            )
    _write_json_atomic(cache_path, cache)
    resolution_fraction = len(out_records) / len(rows)
    if not out_records:
        if invalid_smiles_rows and not unresolved_rows:
            reason = "returned invalid SMILES"
        else:
            reason = "returned no SMILES"
        raise SystemExit(
            f"CosIng PubChem resolution {reason}; resolved 0/{len(rows)} "
            f"rows from {in_csv}. "
            "Use --dry-run only for explicit placeholder diagnostics."
        )
    manifest = {
        "dry_run": False,
        "input_rows": len(rows),
        "resolved_rows": len(out_records),
        "resolution_fraction": resolution_fraction,
        "min_resolution_fraction": args.min_resolution_fraction,
        "min_resolved_count": args.min_resolved_count,
        "unresolved_rows": unresolved_rows[:1000],
        "unresolved_row_count": len(unresolved_rows),
        "invalid_smiles_rows": invalid_smiles_rows[:1000],
        "invalid_smiles_row_count": len(invalid_smiles_rows),
    }
    if (
        len(out_records) < args.min_resolved_count
        or resolution_fraction < args.min_resolution_fraction
    ):
        _write_json_atomic(manifest_path, manifest)
        raise SystemExit(
            "CosIng PubChem resolution did not meet quality gate: "
            f"{len(out_records)}/{len(rows)} resolved "
            f"({resolution_fraction:.3f}); required at least "
            f"{args.min_resolved_count} rows and fraction >= "
            f"{args.min_resolution_fraction:.3f}. "
            f"Rows with no SMILES={len(unresolved_rows)}, "
            f"invalid SMILES={len(invalid_smiles_rows)}"
        )
    _write_parquet_atomic(out_parquet, pd.DataFrame(out_records))
    _write_json_atomic(manifest_path, manifest)
    LOG.info("Wrote cosing.parquet (%d rows)", len(out_records))


if __name__ == "__main__":
    main()
