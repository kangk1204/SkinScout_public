#!/usr/bin/env python3
"""표적 리스트를 받아 SkinScout 검색 인덱스로 발굴한다.

사용자가 관심 표적(바이오마커·유전자·UniProt) 목록을 주면, 표적마다 그 표적에
측정 기록이 있는 화합물을 꺼내고, 그중 화장품 등재 원료(CosIng)와 구조가 같은
것을 표시한다. 결과는 표적×후보 긴 표와, 카테고리별 커버리지 요약으로 나간다.

입력은 CSV/TSV/XLSX. 열 이름은 자동으로 찾고(`--target-column` 등으로 지정 가능),
유전자 기호는 저장소 사전 → `--gene-map` 파일 → `--resolve-online`(UniProt REST)
순으로 해석한다. 리스트에 UniProt 계정번호가 있으면 그대로 쓴다.

예:
  python scripts/discover_from_targets.py --targets targets.csv --out-dir results/discovery/run1
  python scripts/discover_from_targets.py --targets list.xlsx \
      --target-column target --category-column category \
      --gene-map gene_uniprot.csv --top 10
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

POSITIVE_THRESHOLD = 6.0
NEGATIVE_THRESHOLD = 5.0

# UniProt REST: 5xx/429/timeout은 일시 실패로 보고 TTL 뒤 재시도한다. 영구
# null로 캐시하면 다음 실행이 네트워크가 복구되어도 해석을 포기한다.
PERMANENT_HTTP_STATUS = frozenset({400, 404, 410})
TRANSIENT_CACHE_SECONDS = 300.0
ONLINE_LOOKUP_ATTEMPTS = 3
ONLINE_LOOKUP_TIMEOUT = 30.0

ACCESSION_RE = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?:-[0-9]+)?$"
)
TARGET_HEADERS = ("target", "gene", "uniprot", "marker", "biomarker", "표적", "유전자")
CATEGORY_HEADERS = ("category", "카테고리", "분류", "group", "purpose")
DIRECTION_HEADERS = ("direction", "방향")


# ----------------------------------------------------------------- 표적 해석


@dataclass
class TargetRow:
    label: str
    category: str = ""
    direction: str = ""
    gene: str = ""
    uniprot: str = ""
    resolved: bool = False
    reason: str = ""
    candidates: list[str] = field(default_factory=list)
    transient: bool = False


def _norm_header(value: object) -> str:
    return str(value or "").strip().lower()


def _pick_column(columns: list[str], requested: str | None, candidates: tuple[str, ...]) -> str | None:
    if requested:
        for column in columns:
            if _norm_header(column) == _norm_header(requested):
                return column
        raise SystemExit(f"'{requested}' 열을 찾지 못했습니다. 있는 열: {columns}")
    for candidate in candidates:
        for column in columns:
            if _norm_header(column) == _norm_header(candidate):
                return column
    return None


def _load_table(path: Path):
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv", ".txt"}:
        separator = "\t" if suffix in {".tsv", ".txt"} else ","
        return pd.read_csv(path, sep=separator)
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        try:
            import openpyxl  # noqa: F401
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise SystemExit(
                "xlsx를 읽으려면 openpyxl이 필요합니다: python -m pip install openpyxl"
            ) from exc
        raw = pd.read_excel(path, header=None)
        raw = raw.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
        if raw.empty:
            raise SystemExit(f"표가 비어 있습니다: {path}")
        first = [_norm_header(v) for v in raw.iloc[0].tolist()]
        known = set(TARGET_HEADERS) | set(CATEGORY_HEADERS) | set(DIRECTION_HEADERS)
        if any(value in known for value in first):
            raw.columns = [str(v).strip() for v in raw.iloc[0].tolist()]
            return raw.iloc[1:].reset_index(drop=True)
        return raw
    raise SystemExit(f"지원하지 않는 입력 형식입니다: {path}")


def load_target_list(
    path: Path,
    target_column: str | None = None,
    category_column: str | None = None,
    direction_column: str | None = None,
) -> list[TargetRow]:
    frame = _load_table(path)
    columns = [str(c) for c in frame.columns]
    target_col = _pick_column(columns, target_column, TARGET_HEADERS)
    if target_col is None:
        raise SystemExit(f"표적 열을 찾지 못했습니다. 있는 열: {columns}. --target-column 으로 지정하세요.")
    category_col = _pick_column(columns, category_column, CATEGORY_HEADERS)
    direction_col = _pick_column(columns, direction_column, DIRECTION_HEADERS)

    rows: list[TargetRow] = []
    for _, record in frame.iterrows():
        label = str(record.get(target_col) or "").strip()
        if not label:
            continue
        rows.append(
            TargetRow(
                label=label,
                category=str(record.get(category_col) or "").strip() if category_col else "",
                direction=str(record.get(direction_col) or "").strip() if direction_col else "",
            )
        )
    if not rows:
        raise SystemExit(f"표적 행이 없습니다: {path}")
    return rows


def load_gene_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    frame = _load_table(path)
    columns = [str(c) for c in frame.columns]
    gene_col = _pick_column(columns, None, ("gene", "symbol", "유전자"))
    uni_col = _pick_column(columns, None, ("uniprot", "accession", "계정번호"))
    if gene_col is None or uni_col is None:
        raise SystemExit(f"gene-map에는 gene과 uniprot 열이 필요합니다. 있는 열: {columns}")
    return {
        str(record[gene_col]).strip().upper(): str(record[uni_col]).strip()
        for _, record in frame.iterrows()
        if str(record[gene_col]).strip()
    }


def _looks_like_accession(value: str) -> bool:
    return bool(ACCESSION_RE.match(value.strip().upper()))


def _cached_online_lookup(
    cache: dict[str, object], key: str, now: float
) -> tuple[str | None, str, list[str]] | None:
    """Return the cached answer, or None when a fresh request is needed."""
    if key not in cache:
        return None
    cached = cache[key]
    if isinstance(cached, str):
        return cached, "resolved", []
    if cached is None:
        return None, "not_found", []
    if isinstance(cached, dict):
        state = str(cached.get("state") or "")
        candidates = [str(item) for item in cached.get("candidates") or []]
        if state == "ambiguous" and candidates:
            return None, "ambiguous", candidates
        if state == "transient":
            cached_at = float(cached.get("at") or 0.0)
            if now - cached_at < TRANSIENT_CACHE_SECONDS:
                return None, "transient", []
            return None  # TTL이 지났으니 재시도한다.
    return None


def _online_gene_lookup(
    gene: str,
    cache: dict[str, object],
    *,
    now: float | None = None,
    attempts: int = ONLINE_LOOKUP_ATTEMPTS,
    sleep=time.sleep,
) -> tuple[str | None, str, list[str]]:
    """UniProt에서 유전자 기호를 해석한다.

    Returns ``(accession, state, candidates)`` with state one of ``resolved``,
    ``not_found``, ``ambiguous``, ``transient``. A synonym hit alone never
    resolves: gene_exact also matches synonyms (FLG is one for FGFR1), so only
    an exact primary-gene-name match is adopted. Transient failures are cached
    with a TTL instead of a permanent null so a later run retries them.
    """
    key = gene.strip().upper()
    now = time.time() if now is None else now
    cached = _cached_online_lookup(cache, key, now)
    if cached is not None:
        return cached
    query = f"(gene_exact:{key}) AND (organism_id:9606) AND (reviewed:true)"
    url = (
        "https://rest.uniprot.org/uniprotkb/search?query="
        + urllib.parse.quote(query)
        + "&fields=accession,gene_primary&format=json&size=5"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "SkinScout-target-list"})
    results: list[dict] | None = None
    for attempt in range(max(1, attempts)):
        try:
            with urllib.request.urlopen(request, timeout=ONLINE_LOOKUP_TIMEOUT) as response:
                payload = json.loads(response.read().decode("utf-8"))
            results = payload.get("results") or []
            break
        except HTTPError as exc:
            if exc.code in PERMANENT_HTTP_STATUS:
                # 404 등 명확한 거절은 영구 not_found로 캐시한다. 401/403 같은
                # 예상 밖 응답은 TTL 뒤 재시도한다.
                cache[key] = None
                return None, "not_found", []
        except (URLError, TimeoutError, OSError, UnicodeDecodeError, json.JSONDecodeError):
            pass
        if attempt + 1 < max(1, attempts):
            sleep(0.5 * (2**attempt))
    if results is None:
        cache[key] = {"state": "transient", "at": now}
        return None, "transient", []

    exact: set[str] = set()
    synonyms: set[str] = set()
    for result in results:
        accession = result.get("primaryAccession")
        if not accession:
            continue
        primary_names = {
            str((gene_record.get("geneName") or {}).get("value") or "").upper()
            for gene_record in result.get("genes", [])
        }
        (exact if key in primary_names else synonyms).add(str(accession))
    sleep(0.1)
    if len(exact) == 1:
        cache[key] = next(iter(exact))
        return cache[key], "resolved", []
    if exact:
        candidates = sorted(exact)
        cache[key] = {"state": "ambiguous", "candidates": candidates}
        return None, "ambiguous", candidates
    if synonyms:
        # primary가 정확히 일치하는 결과가 없으면 첫 결과를 채택하지 않는다.
        candidates = sorted(synonyms)[:10]
        cache[key] = {"state": "ambiguous", "candidates": candidates}
        return None, "ambiguous", candidates
    cache[key] = None
    return None, "not_found", []


def _candidate_note(candidates: list[str], limit: int = 8) -> str:
    shown = ", ".join(candidates[:limit])
    return shown + ("…" if len(candidates) > limit else "")


def _resolve_one(
    row: TargetRow,
    gene_map: dict[str, str],
    resolve_online: bool,
    online_cache: dict[str, object],
) -> None:
    from explore_target import resolve_target

    label = row.label.strip()
    if _looks_like_accession(label):
        row.uniprot = label.upper()
        row.resolved = True
        row.reason = "UniProt 계정번호"
        return
    # 사용자가 명시한 map이 저장소 사전보다 우선한다: 사용자 입력이 일반
    # 어휘보다 구체적이고, 사전 결과가 충돌하면 사용자가 지정한 값이 이긴다.
    mapped = gene_map.get(label.upper())
    if mapped:
        row.uniprot = mapped
        row.gene = label.upper()
        row.resolved = True
        row.reason = "gene-map"
        return
    entry, candidates = resolve_target(label)
    if entry is not None:
        row.uniprot = entry.uniprot
        row.gene = entry.gene
        row.resolved = True
        row.reason = "저장소 사전"
        return
    if candidates:
        row.candidates = sorted(set(candidates))
        row.reason = (
            f"모호한 표적 이름 (후보 {len(row.candidates)}개: "
            f"{_candidate_note(row.candidates)})"
        )
    if resolve_online:
        accession, state, online_candidates = _online_gene_lookup(label, online_cache)
        if state == "resolved" and accession:
            row.uniprot = accession
            row.gene = label.upper()
            row.resolved = True
            row.reason = "UniProt 조회"
            return
        if state == "ambiguous":
            row.candidates = sorted(set(online_candidates))
            row.reason = (
                f"UniProt gene_exact 동명 표적 {len(row.candidates)}개 "
                f"(primary 유전자 일치 없음: {_candidate_note(row.candidates)})"
            )
            return
        if state == "transient":
            row.transient = True
            row.reason = "UniProt 조회 일시 실패 (TTL 뒤 재시도)"
            return
    if not row.reason:
        row.reason = "해석 실패 (사전·gene-map·온라인 모두 없음)"


def _split_accessions(value: str) -> list[str]:
    return [part for part in re.split(r"[;,\s/]+", value or "") if part]


def resolve_targets(
    rows: list[TargetRow],
    gene_map: dict[str, str] | None = None,
    resolve_online: bool = False,
    online_cache: dict[str, object] | None = None,
) -> list[TargetRow]:
    """한 행이 accession 여러 개를 담으면 각각 별도 표적으로 편다.

    예: "SCF / c-KIT" → KITLG와 KIT. 리간드와 수용체가 한 칸에 적힌 경우가 흔한데,
    하나만 조회하면 다른 쪽에 측정 데이터가 있어도 놓친다.
    """
    gene_map = gene_map or {}
    online_cache = online_cache if online_cache is not None else {}
    output: list[TargetRow] = []

    for row in rows:
        parts = _split_accessions(row.label)
        if len(parts) > 1 and all(_looks_like_accession(part) for part in parts):
            row.uniprot = ";".join(part.upper() for part in parts)
            row.resolved = True
            row.reason = "UniProt 계정번호"
        else:
            _resolve_one(row, gene_map, resolve_online, online_cache)

        accessions = _split_accessions(row.uniprot) if row.resolved else []
        if len(accessions) > 1:
            row.uniprot = accessions[0]
            output.append(row)
            for extra in accessions[1:]:
                output.append(
                    TargetRow(
                        label=row.label,
                        category=row.category,
                        direction=row.direction,
                        gene=row.gene,
                        uniprot=extra,
                        resolved=True,
                        reason=row.reason,
                    )
                )
        else:
            output.append(row)
    return output


def _threshold_label(max_pactivity: object) -> str:
    try:
        value = float(max_pactivity)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "unknown"
    if value >= POSITIVE_THRESHOLD:
        return "positive"
    if value >= NEGATIVE_THRESHOLD:
        return "borderline"
    return "below"


def discover(
    index_dir: Path,
    rows: list[TargetRow],
    top: int,
    mode: str,
    edges=None,
    ligands=None,
) -> tuple[list[dict], list[dict]]:
    from explore_target import load_index, rank_target

    if edges is None or ligands is None:
        edges, ligands = load_index(index_dir)
    index_targets = set(edges["uniprot"].unique())
    candidates: list[dict] = []
    coverage: list[dict] = []

    for row in rows:
        record = {
            "category": row.category,
            "target_label": row.label,
            "gene": row.gene,
            "uniprot": row.uniprot,
            "direction": row.direction,
            "resolved": row.resolved,
            "in_index": False,
            "reason": row.reason,
            "candidate_uniprots": ";".join(row.candidates),
            "transient_failure": row.transient,
            "n_candidates": 0,
            "n_registered": 0,
        }
        if row.resolved and row.uniprot in index_targets:
            record["in_index"] = True
            record["reason"] = "인덱스에 있음"
            frame = rank_target(edges, ligands, row.uniprot, top, mode)
            record["n_candidates"] = int(len(frame))
            for rank, (_, hit) in enumerate(frame.iterrows(), 1):
                candidates.append(
                    {
                        "category": row.category,
                        "target_label": row.label,
                        "gene": row.gene or row.uniprot,
                        "uniprot": row.uniprot,
                        "direction": row.direction,
                        "rank": rank,
                        "smiles": str(hit.get("canonical_smiles") or ""),
                        "inchikey": str(hit.get("standard_inchikey") or ""),
                        "max_pactivity": hit.get("max_pactivity"),
                        "median_pactivity": hit.get("median_pactivity"),
                        "positive_measurement_count": hit.get("positive_measurement_count"),
                        "publication_count": hit.get("publication_count"),
                        "measurement_count": hit.get("measurement_count"),
                        "source_db": str(hit.get("source_db") or ""),
                        "threshold": _threshold_label(hit.get("max_pactivity")),
                        "cosing_match": "none",
                        "inci_name": "",
                    }
                )
        elif row.resolved:
            record["reason"] = "인덱스 4,873 표적 밖 (측정 데이터 없음)"
        coverage.append(record)
    return candidates, coverage


def cosing_index(frame) -> tuple[set[str], set[str], dict[str, str]]:
    """Return exact/connectivity keys and names indexed by each identity level."""
    exact: set[str] = set()
    skeletons: set[str] = set()
    name_sets: dict[str, set[str]] = {}
    for _, record in frame.fillna("").iterrows():
        key = str(record.get("inchikey") or "").strip()
        if not key:
            continue
        exact.add(key)
        skeleton = str(record.get("skeleton") or key[:14])
        skeletons.add(skeleton)
        name = str(record.get("inci_name") or "").strip()
        for identity in (key, skeleton):
            name_sets.setdefault(identity, set())
            if name:
                name_sets[identity].add(name)
    names = {identity: "; ".join(sorted(values)) for identity, values in name_sets.items()}
    return exact, skeletons, names


def annotate_cosing(candidates: list[dict], frame) -> int:
    if frame is None or getattr(frame, "empty", True):
        return 0
    exact, skeletons, names = cosing_index(frame)
    registered: set[str] = set()
    for candidate in candidates:
        key = candidate["inchikey"]
        if key and key in exact:
            candidate["cosing_match"] = "exact"
            candidate["inci_name"] = names.get(key, "")
            registered.add(candidate["uniprot"])
        elif key and key[:14] in skeletons:
            candidate["cosing_match"] = "connectivity"
            candidate["inci_name"] = names.get(key[:14], "")
    return len(registered)


def write_reports(
    out_dir: Path,
    rows: list[TargetRow],
    candidates: list[dict],
    coverage: list[dict],
    top: int,
    mode: str,
    index_dir: Path,
    smiles_out: Path | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_fields = list(candidates[0].keys()) if candidates else [
        "category", "target_label", "gene", "uniprot", "direction", "rank", "smiles",
        "inchikey", "max_pactivity", "median_pactivity", "positive_measurement_count",
        "publication_count", "measurement_count", "source_db", "threshold",
        "cosing_match", "inci_name",
    ]
    with (out_dir / "discovery_candidates.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=candidate_fields)
        writer.writeheader()
        writer.writerows(candidates)

    coverage_fields = list(coverage[0].keys()) if coverage else []
    with (out_dir / "target_coverage.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=coverage_fields)
        writer.writeheader()
        writer.writerows(coverage)

    with (out_dir / "unresolved.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["category", "target_label", "reason"])
        writer.writeheader()
        for record in coverage:
            if not record["in_index"]:
                writer.writerow(
                    {
                        "category": record["category"],
                        "target_label": record["target_label"],
                        "reason": record["reason"],
                    }
                )

    if smiles_out is not None:
        seen: set[str] = set()
        lines: list[str] = []
        for candidate in candidates:
            smiles = candidate["smiles"]
            if smiles and smiles not in seen:
                seen.add(smiles)
                lines.append(smiles)
        smiles_out.parent.mkdir(parents=True, exist_ok=True)
        smiles_out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    total = len(coverage)
    resolved = sum(1 for record in coverage if record["resolved"])
    covered = sum(1 for record in coverage if record["in_index"])
    registered = {(c["uniprot"], c["inchikey"]) for c in candidates if c["cosing_match"] == "exact"}

    lines = [
        "# 표적 리스트 발굴 요약",
        "",
        f"- 입력 표적: **{total}**",
        f"- 이름 해석: **{resolved}** / 인덱스 조회 가능: **{covered}**",
        f"- 후보 화합물(중복 포함): **{len(candidates)}**, 그중 CosIng 등재 원료와 구조 일치: **{len(registered)}**",
        f"- 정렬 `{mode}`, 표적당 상위 {top}개",
        "",
        "> 여기서 '알려진'은 ChEMBL·BindingDB에 측정 기록이 있다는 뜻입니다. "
        "\"측정됨\"이 \"세다\"와 다르므로 `max_pactivity`와 `threshold`를 함께 보세요 "
        "(양성 6.0, 경계 5.0). 이 표는 새 결합을 예측한 것이 아니라 측정 기록을 정리한 것입니다.",
        "",
    ]
    if smiles_out is not None:
        lines.append(f"- 파이프라인 입력용 SMILES: `{smiles_out}`")
        lines.append("")

    categories: list[str] = []
    for record in coverage:
        if record["category"] and record["category"] not in categories:
            categories.append(record["category"])
    categories += [""] if any(not record["category"] for record in coverage) else []

    for category in categories:
        group = [record for record in coverage if record["category"] == category]
        title = category or "(카테고리 없음)"
        lines.append(f"## {title}")
        lines.append("")
        lines.append(
            f"- 표적 {len(group)}개 · 해석 {sum(1 for r in group if r['resolved'])}개 · "
            f"인덱스 조회 {sum(1 for r in group if r['in_index'])}개"
        )
        hits = [c for c in candidates if c["category"] == category]
        if hits:
            registered_hits = [c for c in hits if c["cosing_match"] == "exact"]
            if registered_hits:
                lines.append("- 등재 원료와 구조 일치 후보:")
                for candidate in registered_hits[:5]:
                    lines.append(
                        f"  - {candidate['inci_name'] or candidate['inchikey'][:14]} "
                        f"({candidate['gene']}, pAct {candidate['max_pactivity']}, {candidate['threshold']})"
                    )
            else:
                lines.append("- 등재 원료와 구조가 같은 후보는 없습니다. 아래 후보는 측정 근거용입니다.")
        else:
            lines.append("- 조회 가능한 표적이 없어 후보가 없습니다.")
        lines.append("")

    lines += [
        "## 다음 단계",
        "",
        "1. 등재 원료 일치 후보를 `python scripts/run_skinscout.py --smiles '<SMILES>' --mode fast`로 "
        "안전성·표적 분석에 넣습니다.",
        "2. 등재 원료 일치가 없으면 `scripts/explore_target.py <UniProt>`로 표적별 후보를 직접 보고, "
        "`scripts/alternative_ingredients.py --smiles '<활성 화합물>'`로 핵심구조 유지 대체 원료를 찾습니다.",
        "3. 실험 전 가설입니다. `근거 유사도`와 `threshold`를 함께 읽고, assay로 확인하세요.",
        "",
        f"인덱스: `{index_dir}`",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", required=True, type=Path, help="표적 리스트 (CSV/TSV/XLSX)")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--target-column")
    parser.add_argument("--category-column")
    parser.add_argument("--direction-column")
    parser.add_argument("--gene-map", type=Path, help="gene,uniprot 매핑 파일 (오프라인 해석)")
    parser.add_argument("--resolve-online", action="store_true", help="사전에 없는 기호를 UniProt REST로 해석")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--mode", choices=("balanced", "potency", "evidence"), default="balanced")
    parser.add_argument("--index-dir", type=Path)
    parser.add_argument("--smiles-out", type=Path, help="상위 후보 SMILES 목록 저장")
    parser.add_argument("--no-cosing", action="store_true", help="CosIng 원료 대조를 건너뛴다")
    args = parser.parse_args()

    if args.top < 1:
        parser.error("--top 은 1 이상이어야 합니다")

    from explore_target import _discover_index_dir

    index_dir = args.index_dir or _discover_index_dir()
    if index_dir is None:
        raise SystemExit("검색 인덱스가 없습니다. Stage 0 표적 데이터를 먼저 준비하세요.")

    rows = load_target_list(args.targets, args.target_column, args.category_column, args.direction_column)
    gene_map = load_gene_map(args.gene_map)
    cache_path = args.out_dir / "uniprot_cache.json"
    online_cache: dict[str, object] = {}
    if cache_path.is_file():
        try:
            online_cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            online_cache = {}
    rows = resolve_targets(rows, gene_map, args.resolve_online, online_cache)
    if args.resolve_online and online_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(online_cache, ensure_ascii=False, indent=1), encoding="utf-8")

    candidates, coverage = discover(index_dir, rows, args.top, args.mode)

    if not args.no_cosing:
        try:
            from alternative_ingredients import load_ingredient_library

            library = load_ingredient_library()
            annotate_cosing(candidates, library.frame)
        except (FileNotFoundError, ImportError) as exc:
            print(f"[discover] CosIng 대조를 건너뜁니다: {exc}", file=sys.stderr)
            for candidate in candidates:
                candidate["cosing_match"] = "unavailable"

    for record in coverage:
        record["n_registered"] = sum(
            1
            for candidate in candidates
            if candidate["uniprot"] == record["uniprot"] and candidate["cosing_match"] == "exact"
        )

    write_reports(
        args.out_dir, rows, candidates, coverage, args.top, args.mode, index_dir, args.smiles_out
    )

    resolved = sum(1 for row in rows if row.resolved)
    covered = sum(1 for record in coverage if record["in_index"])
    registered = sum(1 for candidate in candidates if candidate["cosing_match"] == "exact")
    print(
        f"표적 {len(rows)} | 해석 {resolved} | 조회 가능 {covered} | "
        f"후보 {len(candidates)} | 등재 일치 {registered}"
    )
    print(f"리포트: {args.out_dir}/summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
