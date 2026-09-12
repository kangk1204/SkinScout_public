#!/usr/bin/env python3
"""Mirror the public EC CosIng search API into the Stage 0 source CSV.

The EC search API caps a single query window at roughly 10,000 hits. This
mirror splits the INCI-name search into prefix windows, deduplicates records by
the official CosIng reference identifiers, and fails closed when the collected
unique row count does not match the API-reported total.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import requests


OFFICIAL_SEARCH_API_URL = "https://api.tech.ec.europa.eu/search-api/prod/rest/search"
# Public key shipped by the official EC CosIng single-page app config.
OFFICIAL_SEARCH_API_KEY = "285a77fd-1257-4271-8507-f0c6b2961203"
SEARCH_FIELDS = ["inciName.exact"]
DEFAULT_PREFIX_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
# The public API tokenizes INCI names containing Greek letters separately from
# ASCII windows. Keep prefix windows stable, but include Greek residual suffix
# probes so official rows such as alpha/beta stereochemistry are not missed.
DEFAULT_RESIDUAL_SUFFIX_ALPHABET = (
    DEFAULT_PREFIX_ALPHABET
    + "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
    + "αβγδεζηθικλμνξοπρστυφχψω"
    + "µμ"
)
CSV_COLUMNS = [
    "INCI name",
    "CAS",
    "EINECS",
    "Function",
    "Item type",
    "Substance ID",
    "Reference",
    "Current version",
]


class MirrorError(RuntimeError):
    """Raised when CosIng mirroring cannot produce a complete source CSV."""


def _log(message: str) -> None:
    print(f"[stage0.cosing.mirror] {message}", file=sys.stderr, flush=True)


@dataclass(frozen=True)
class SearchPage:
    query_text: str
    page_number: int
    total_results: int
    warnings: list[str]
    results: list[Mapping[str, Any]]


@dataclass(frozen=True)
class PrefixWindow:
    query_text: str
    total_results: int
    depth: int


def _remove_outputs(*paths: Path) -> None:
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _write_csv_atomic(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _query_payload(query_text: str) -> dict[str, Any]:
    return {
        "bool": {
            "must": [
                {
                    "text": {
                        "query": query_text,
                        "fields": SEARCH_FIELDS,
                        "defaultOperator": "AND",
                        "analyzeWildcard": True,
                    }
                }
            ]
        }
    }


def _post_search(
    *,
    session: requests.Session,
    api_url: str,
    api_key: str,
    query_text: str,
    page_size: int,
    page_number: int,
    timeout_s: float,
) -> SearchPage:
    response = session.post(
        api_url,
        params={
            "apiKey": api_key,
            "text": query_text,
            "pageSize": str(page_size),
            "pageNumber": str(page_number),
        },
        files={
            "query": (
                "query.json",
                json.dumps(_query_payload(query_text)),
                "application/json",
            )
        },
        timeout=timeout_s,
    )
    status_code = int(getattr(response, "status_code", 200))
    if status_code >= 400:
        text = str(getattr(response, "text", ""))[:500]
        raise MirrorError(
            f"CosIng API request failed for {query_text!r} page {page_number}: "
            f"HTTP {status_code} {text}"
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise MirrorError(
            f"CosIng API returned non-JSON response for {query_text!r} "
            f"page {page_number}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise MirrorError(
            f"CosIng API returned non-object response for {query_text!r} "
            f"page {page_number}"
        )
    total = int(payload.get("totalResults") or 0)
    warnings = [str(item) for item in payload.get("warnings") or []]
    results = payload.get("results") or []
    if not isinstance(results, list):
        raise MirrorError(
            f"CosIng API results field is not a list for {query_text!r} "
            f"page {page_number}"
        )
    return SearchPage(
        query_text=query_text,
        page_number=page_number,
        total_results=total,
        warnings=warnings,
        results=[item for item in results if isinstance(item, Mapping)],
    )


def _post_search_with_retries(
    *,
    session: requests.Session,
    api_url: str,
    api_key: str,
    query_text: str,
    page_size: int,
    page_number: int,
    timeout_s: float,
    max_retries: int,
    retry_backoff_s: float,
) -> SearchPage:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            page = _post_search(
                session=session,
                api_url=api_url,
                api_key=api_key,
                query_text=query_text,
                page_size=page_size,
                page_number=page_number,
                timeout_s=timeout_s,
            )
        except (MirrorError, requests.RequestException) as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            _log(
                f"retry {query_text} page {page_number} after request error "
                f"({attempt + 1}/{max_retries}): {exc}"
            )
        else:
            if not page.warnings or attempt >= max_retries:
                return page
            _log(
                f"retry {query_text} page {page_number} after API warnings "
                f"({attempt + 1}/{max_retries}): {page.warnings}"
            )
        if retry_backoff_s:
            time.sleep(retry_backoff_s * (attempt + 1))
    if last_error is not None:
        raise MirrorError(
            f"CosIng API request failed for {query_text!r} page {page_number} "
            f"after {max_retries + 1} attempts: {last_error}"
        ) from last_error
    raise MirrorError(
        f"CosIng API kept returning warnings for {query_text!r} page "
        f"{page_number} after {max_retries + 1} attempts"
    )


def _discover_prefix_windows(
    *,
    session: requests.Session,
    api_url: str,
    api_key: str,
    page_size: int,
    max_window_results: int,
    max_prefix_depth: int,
    prefix_alphabet: str,
    timeout_s: float,
    sleep_s: float,
    max_retries: int,
    retry_backoff_s: float,
) -> list[PrefixWindow]:
    windows: list[PrefixWindow] = []

    def visit(prefix: str) -> None:
        query_text = f"{prefix}*"
        page = _post_search_with_retries(
            session=session,
            api_url=api_url,
            api_key=api_key,
            query_text=query_text,
            page_size=1,
            page_number=1,
            timeout_s=timeout_s,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
        )
        if sleep_s:
            time.sleep(sleep_s)
        if len(prefix) == 1:
            _log(f"sized prefix {query_text}: total={page.total_results}")
        if page.total_results == 0:
            return
        if page.warnings:
            raise MirrorError(
                f"CosIng API returned warnings while sizing {query_text!r}: "
                f"{page.warnings}"
            )
        if page.total_results > max_window_results:
            if len(prefix) >= max_prefix_depth:
                raise MirrorError(
                    f"CosIng prefix window {query_text!r} has "
                    f"{page.total_results} hits, above max_window_results="
                    f"{max_window_results}; increase --max-prefix-depth"
                )
            _log(
                f"splitting prefix {query_text}: total={page.total_results}, "
                f"depth={len(prefix)}/{max_prefix_depth}"
            )
            for char in prefix_alphabet:
                visit(prefix + char)
            return
        windows.append(
            PrefixWindow(
                query_text=query_text,
                total_results=page.total_results,
                depth=len(prefix),
            )
        )
        if len(prefix) > 1:
            _log(
                f"accepted prefix window {query_text}: "
                f"total={page.total_results}, depth={len(prefix)}"
            )

    for char in prefix_alphabet:
        visit(char)
    return windows


def _discover_suffix_windows(
    *,
    session: requests.Session,
    api_url: str,
    api_key: str,
    page_size: int,
    max_window_results: int,
    max_suffix_depth: int,
    suffix_alphabet: str,
    timeout_s: float,
    sleep_s: float,
    max_retries: int,
    retry_backoff_s: float,
) -> list[PrefixWindow]:
    windows: list[PrefixWindow] = []

    def visit(suffix: str) -> None:
        query_text = f"*{suffix}"
        page = _post_search_with_retries(
            session=session,
            api_url=api_url,
            api_key=api_key,
            query_text=query_text,
            page_size=1,
            page_number=1,
            timeout_s=timeout_s,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
        )
        if sleep_s:
            time.sleep(sleep_s)
        if len(suffix) == 1:
            _log(f"sized suffix {query_text}: total={page.total_results}")
        if page.total_results == 0:
            return
        if page.warnings:
            raise MirrorError(
                f"CosIng API returned warnings while sizing suffix "
                f"{query_text!r}: {page.warnings}"
            )
        if page.total_results > max_window_results:
            if len(suffix) >= max_suffix_depth:
                raise MirrorError(
                    f"CosIng suffix window {query_text!r} has "
                    f"{page.total_results} hits, above max_window_results="
                    f"{max_window_results}; increase --max-suffix-depth"
                )
            _log(
                f"splitting suffix {query_text}: total={page.total_results}, "
                f"depth={len(suffix)}/{max_suffix_depth}"
            )
            for char in suffix_alphabet:
                visit(char + suffix)
            return
        windows.append(
            PrefixWindow(
                query_text=query_text,
                total_results=page.total_results,
                depth=len(suffix),
            )
        )
        if len(suffix) > 1:
            _log(
                f"accepted suffix window {query_text}: "
                f"total={page.total_results}, depth={len(suffix)}"
            )

    for char in suffix_alphabet:
        visit(char)
    windows.sort(key=lambda window: (window.total_results, window.query_text))
    return windows


def _result_metadata(result: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = result.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    merged = dict(metadata)
    search_reference = result.get("reference")
    if search_reference:
        merged["__search_reference"] = [str(search_reference)]
    return merged


def _values(metadata: Mapping[str, Any], key: str) -> list[str]:
    raw = metadata.get(key)
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        values = raw
    else:
        values = [raw]
    return [str(value).strip() for value in values if str(value).strip()]


def _first(metadata: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        values = _values(metadata, key)
        if values:
            return values[0]
    return ""


def _join(metadata: Mapping[str, Any], key: str) -> str:
    seen: dict[str, None] = {}
    for value in _values(metadata, key):
        if value != "-":
            seen.setdefault(value, None)
    return ";".join(seen)


def _record_key(metadata: Mapping[str, Any]) -> str:
    # The public API can expose multiple display rows under the same reference
    # or substance identifier. Use the stable identifier fields plus displayed
    # INCI/function metadata so overlap windows dedupe exact rows without
    # collapsing distinct official rows.
    parts = [
        _first(metadata, "__search_reference"),
        _first(metadata, "reference"),
        _first(metadata, "REFERENCE"),
        _first(metadata, "substanceId"),
        _first(metadata, "refNo"),
        _first(metadata, "currentVersion"),
        _first(metadata, "inciName", "inciUsaName", "innName", "phEurName"),
        _join(metadata, "casNo"),
        _join(metadata, "ecNo"),
        _join(metadata, "functionName"),
        _join(metadata, "itemType"),
    ]
    key = "|".join(parts)
    if key.strip("|"):
        return f"record:{key}"
    raise MirrorError("CosIng result lacks identifiers and INCI metadata")

def _metadata_to_csv_row(metadata: Mapping[str, Any]) -> dict[str, str] | None:
    inci_name = _first(metadata, "inciName", "inciUsaName", "innName", "phEurName")
    if not inci_name:
        return None
    return {
        "INCI name": inci_name,
        "CAS": _join(metadata, "casNo"),
        "EINECS": _join(metadata, "ecNo"),
        "Function": _join(metadata, "functionName"),
        "Item type": _join(metadata, "itemType"),
        "Substance ID": _first(metadata, "substanceId"),
        "Reference": _first(metadata, "reference"),
        "Current version": _first(metadata, "currentVersion"),
    }


def _fetch_window_records(
    *,
    session: requests.Session,
    api_url: str,
    api_key: str,
    window: PrefixWindow,
    page_size: int,
    timeout_s: float,
    sleep_s: float,
    progress_every_pages: int,
    max_retries: int,
    retry_backoff_s: float,
) -> Iterable[Mapping[str, Any]]:
    pages = math.ceil(window.total_results / page_size)
    for page_number in range(1, pages + 1):
        page = _post_search_with_retries(
            session=session,
            api_url=api_url,
            api_key=api_key,
            query_text=window.query_text,
            page_size=page_size,
            page_number=page_number,
            timeout_s=timeout_s,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
        )
        if page.warnings:
            raise MirrorError(
                f"CosIng API returned warnings for {window.query_text!r} "
                f"page {page_number}: {page.warnings}"
            )
        if page.total_results != window.total_results:
            raise MirrorError(
                f"CosIng API total changed for {window.query_text!r}: "
                f"{window.total_results} -> {page.total_results}"
            )
        if (
            progress_every_pages > 0
            and (
                page_number == 1
                or page_number == pages
                or page_number % progress_every_pages == 0
            )
        ):
            rows_seen = min(
                (page_number - 1) * page_size + len(page.results),
                window.total_results,
            )
            _log(
                f"fetch {window.query_text}: page {page_number}/{pages}, "
                f"rows_seen={rows_seen}/{window.total_results}"
            )
        for result in page.results:
            yield _result_metadata(result)
        if sleep_s:
            time.sleep(sleep_s)


def mirror_cosing_api(
    *,
    out_dir: Path,
    manifest: Path | None = None,
    api_url: str = OFFICIAL_SEARCH_API_URL,
    api_key: str = OFFICIAL_SEARCH_API_KEY,
    page_size: int = 200,
    max_window_results: int = 10000,
    max_prefix_depth: int = 3,
    max_suffix_depth: int = 3,
    prefix_alphabet: str = DEFAULT_PREFIX_ALPHABET,
    residual_suffix_alphabet: str = DEFAULT_RESIDUAL_SUFFIX_ALPHABET,
    timeout_s: float = 60.0,
    sleep_s: float = 0.01,
    progress_every_pages: int = 10,
    max_retries: int = 5,
    retry_backoff_s: float = 2.0,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    if page_size <= 0 or page_size > 200:
        raise MirrorError("--page-size must be between 1 and 200")
    if max_window_results <= 0:
        raise MirrorError("--max-window-results must be positive")
    if max_prefix_depth <= 0:
        raise MirrorError("--max-prefix-depth must be positive")
    if max_suffix_depth <= 0:
        raise MirrorError("--max-suffix-depth must be positive")
    if progress_every_pages < 0:
        raise MirrorError("--progress-every-pages must be non-negative")
    if max_retries < 0:
        raise MirrorError("--max-retries must be non-negative")
    if retry_backoff_s < 0:
        raise MirrorError("--retry-backoff-s must be non-negative")

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "cosing.csv"
    manifest_path = manifest or out_dir / "cosing_api_manifest.json"
    _remove_outputs(csv_path, manifest_path)

    client = session or requests.Session()
    try:
        _log(
            "starting EC CosIng public API mirror: "
            f"page_size={page_size}, max_window_results={max_window_results}, "
            f"max_prefix_depth={max_prefix_depth}"
        )
        expected_page = _post_search_with_retries(
            session=client,
            api_url=api_url,
            api_key=api_key,
            query_text="***",
            page_size=1,
            page_number=1,
            timeout_s=timeout_s,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
        )
        if expected_page.warnings:
            raise MirrorError(
                f"CosIng API returned warnings while sizing full mirror: "
                f"{expected_page.warnings}"
            )
        expected_total = expected_page.total_results
        if expected_total <= 0:
            raise MirrorError("CosIng API returned zero total records for full mirror")
        _log(f"official full-query total: {expected_total}")

        windows = _discover_prefix_windows(
            session=client,
            api_url=api_url,
            api_key=api_key,
            page_size=page_size,
            max_window_results=max_window_results,
            max_prefix_depth=max_prefix_depth,
            prefix_alphabet=prefix_alphabet,
            timeout_s=timeout_s,
            sleep_s=sleep_s,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
        )
        if not windows:
            raise MirrorError("CosIng prefix discovery found no result windows")
        window_total = sum(window.total_results for window in windows)
        max_window = max(windows, key=lambda window: window.total_results)
        _log(
            f"discovered {len(windows)} prefix windows: "
            f"window_result_sum={window_total}, largest={max_window.query_text} "
            f"({max_window.total_results})"
        )

        records: dict[str, dict[str, str]] = {}
        prefix_stats: list[dict[str, Any]] = []
        total_fetched = 0
        residual_fetch_total = 0
        residual_new_unique_rows = 0
        residual_full_query_error = ""
        suffix_stats: list[dict[str, Any]] = []
        for window_index, window in enumerate(windows, start=1):
            before = len(records)
            fetched = 0
            _log(
                f"window {window_index}/{len(windows)} {window.query_text}: "
                f"total={window.total_results}, unique_before={before}"
            )
            for metadata in _fetch_window_records(
                session=client,
                api_url=api_url,
                api_key=api_key,
                window=window,
                page_size=page_size,
                timeout_s=timeout_s,
                sleep_s=sleep_s,
                progress_every_pages=progress_every_pages,
                max_retries=max_retries,
                retry_backoff_s=retry_backoff_s,
            ):
                fetched += 1
                row = _metadata_to_csv_row(metadata)
                if row is None:
                    continue
                records.setdefault(_record_key(metadata), row)
            prefix_stats.append(
                {
                    **asdict(window),
                    "fetched_rows": fetched,
                    "new_unique_rows": len(records) - before,
                }
            )
            total_fetched += fetched
            _log(
                f"window {window_index}/{len(windows)} {window.query_text} done: "
                f"fetched={fetched}, new_unique={len(records) - before}, "
                f"unique_total={len(records)}"
            )

        if len(records) != expected_total:
            before = len(records)
            _log(
                "prefix collection short of official total "
                f"({before}/{expected_total}); fetching full-query residual pages"
            )
            full_window = PrefixWindow(
                query_text="***",
                total_results=expected_total,
                depth=0,
            )
            try:
                for metadata in _fetch_window_records(
                    session=client,
                    api_url=api_url,
                    api_key=api_key,
                    window=full_window,
                    page_size=page_size,
                    timeout_s=timeout_s,
                    sleep_s=sleep_s,
                    progress_every_pages=progress_every_pages,
                    max_retries=max_retries,
                    retry_backoff_s=retry_backoff_s,
                ):
                    residual_fetch_total += 1
                    row = _metadata_to_csv_row(metadata)
                    if row is None:
                        continue
                    records.setdefault(_record_key(metadata), row)
                    if len(records) == expected_total:
                        break
            except MirrorError as exc:
                residual_full_query_error = str(exc)
                _log(
                    "full-query residual stopped before completion; "
                    f"falling back to suffix windows: {exc}"
                )
            residual_new_unique_rows = len(records) - before
            _log(
                "full-query residual fetch done: "
                f"fetched={residual_fetch_total}, "
                f"new_unique={residual_new_unique_rows}, "
                f"unique_total={len(records)}"
            )

        if len(records) != expected_total and residual_suffix_alphabet:
            before = len(records)
            _log(
                "full-query residual still short of official total "
                f"({before}/{expected_total}); discovering suffix residual windows"
            )
            suffix_windows = _discover_suffix_windows(
                session=client,
                api_url=api_url,
                api_key=api_key,
                page_size=page_size,
                max_window_results=max_window_results,
                max_suffix_depth=max_suffix_depth,
                suffix_alphabet=residual_suffix_alphabet,
                timeout_s=timeout_s,
                sleep_s=sleep_s,
                max_retries=max_retries,
                retry_backoff_s=retry_backoff_s,
            )
            if not suffix_windows:
                raise MirrorError("CosIng suffix residual discovery found no windows")
            _log(
                f"discovered {len(suffix_windows)} suffix residual windows: "
                f"window_result_sum={sum(w.total_results for w in suffix_windows)}"
            )
            for window_index, window in enumerate(suffix_windows, start=1):
                if len(records) == expected_total:
                    break
                suffix_before = len(records)
                fetched = 0
                _log(
                    f"suffix window {window_index}/{len(suffix_windows)} "
                    f"{window.query_text}: total={window.total_results}, "
                    f"unique_before={suffix_before}"
                )
                for metadata in _fetch_window_records(
                    session=client,
                    api_url=api_url,
                    api_key=api_key,
                    window=window,
                    page_size=page_size,
                    timeout_s=timeout_s,
                    sleep_s=sleep_s,
                    progress_every_pages=progress_every_pages,
                    max_retries=max_retries,
                    retry_backoff_s=retry_backoff_s,
                ):
                    fetched += 1
                    row = _metadata_to_csv_row(metadata)
                    if row is None:
                        continue
                    records.setdefault(_record_key(metadata), row)
                    if len(records) == expected_total:
                        break
                suffix_stats.append(
                    {
                        **asdict(window),
                        "fetched_rows": fetched,
                        "new_unique_rows": len(records) - suffix_before,
                    }
                )
                _log(
                    f"suffix window {window_index}/{len(suffix_windows)} "
                    f"{window.query_text} done: fetched={fetched}, "
                    f"new_unique={len(records) - suffix_before}, "
                    f"unique_total={len(records)}"
                )

        if len(records) != expected_total:
            raise MirrorError(
                "CosIng mirror incomplete: expected "
                f"{expected_total} unique records from the official API, "
                f"collected {len(records)} after prefix dedupe"
            )

        rows = sorted(records.values(), key=lambda row: row["INCI name"].upper())
        _write_csv_atomic(csv_path, rows)
        manifest_payload: dict[str, Any] = {
            "source": "EC CosIng public search API",
            "api_url": api_url,
            "search_fields": SEARCH_FIELDS,
            "generated_at": datetime.now(UTC).isoformat(),
            "expected_total": expected_total,
            "unique_records": len(rows),
            "page_size": page_size,
            "max_window_results": max_window_results,
            "max_prefix_depth": max_prefix_depth,
            "max_suffix_depth": max_suffix_depth,
            "overlap_fetch_total": total_fetched,
            "overlap_duplicate_or_skipped_rows": total_fetched - len(rows),
            "residual_full_query_fetched_rows": residual_fetch_total,
            "residual_full_query_new_unique_rows": residual_new_unique_rows,
            "residual_full_query_error": residual_full_query_error,
            "residual_suffix_windows": suffix_stats,
            "prefix_windows": prefix_stats,
            "output_csv": str(csv_path),
        }
        _write_json_atomic(manifest_path, manifest_payload)
        _log(f"mirror complete: unique_records={len(rows)}, output={csv_path}")
        return manifest_payload
    except Exception:
        _remove_outputs(csv_path, manifest_path)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--api-url", default=OFFICIAL_SEARCH_API_URL)
    parser.add_argument("--api-key", default=OFFICIAL_SEARCH_API_KEY)
    parser.add_argument("--page-size", type=int, default=200)
    parser.add_argument("--max-window-results", type=int, default=10000)
    parser.add_argument("--max-prefix-depth", type=int, default=3)
    parser.add_argument("--max-suffix-depth", type=int, default=3)
    parser.add_argument("--prefix-alphabet", default=DEFAULT_PREFIX_ALPHABET)
    parser.add_argument(
        "--residual-suffix-alphabet",
        default=DEFAULT_RESIDUAL_SUFFIX_ALPHABET,
        help="alphabet used for suffix residual windows after prefix mirroring",
    )
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--sleep-s", type=float, default=0.01)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-backoff-s", type=float, default=2.0)
    parser.add_argument(
        "--progress-every-pages",
        type=int,
        default=10,
        help="log fetch progress every N pages; use 0 to disable page progress",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = mirror_cosing_api(
            out_dir=args.out_dir,
            manifest=args.manifest,
            api_url=args.api_url,
            api_key=args.api_key,
            page_size=args.page_size,
            max_window_results=args.max_window_results,
            max_prefix_depth=args.max_prefix_depth,
            max_suffix_depth=args.max_suffix_depth,
            prefix_alphabet=args.prefix_alphabet,
            residual_suffix_alphabet=args.residual_suffix_alphabet,
            timeout_s=args.timeout_s,
            sleep_s=args.sleep_s,
            progress_every_pages=args.progress_every_pages,
            max_retries=args.max_retries,
            retry_backoff_s=args.retry_backoff_s,
        )
    except MirrorError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        "Mirrored EC CosIng API: "
        f"{manifest['unique_records']} records -> {manifest['output_csv']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
