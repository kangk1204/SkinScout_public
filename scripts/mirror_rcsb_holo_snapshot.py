#!/usr/bin/env python3
"""Mirror an RCSB PDB human holo positive-only contact snapshot.

This source is intended only for strict known compound-protein recovery
evaluation. It must not be used for training, calibration, or model selection.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import requests


SCHEMA_VERSION = "skinscout.rcsb-holo-contact-snapshot.v1"
SEARCH_API_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
DATA_API_GRAPHQL_URL = "https://data.rcsb.org/graphql"
RCSB_CC0_LICENSE_URL = "https://www.rcsb.org/pages/policies"
SOURCE_NAME = "RCSB PDB"
DEFAULT_RELEASE_CUTOFF = "2025-01-01"
DEFAULT_MIN_NONPOLYMER_MW = 50.0

SEARCH_QUERY_CONTRACT = {
    "query": {
        "type": "group",
        "logical_operator": "and",
        "nodes": [
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "exptl.method",
                    "operator": "exists",
                },
            },
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_accession_info.initial_release_date",
                    "operator": "greater_or_equal",
                    "value": DEFAULT_RELEASE_CUTOFF,
                },
            },
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_entity_source_organism.taxonomy_lineage.id",
                    "operator": "exact_match",
                    "value": "9606",
                },
            },
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "chem_comp.formula_weight",
                    "operator": "greater",
                    "value": DEFAULT_MIN_NONPOLYMER_MW,
                },
            },
            {
                "type": "group",
                "logical_operator": "or",
                "nodes": [
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_nonpolymer_entity_annotation.type",
                            "operator": "exact_match",
                            "value": "SUBJECT_OF_INVESTIGATION",
                        },
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_nonpolymer_instance_annotation.type",
                            "operator": "exact_match",
                            "value": "SUBJECT_OF_INVESTIGATION",
                        },
                    },
                ],
            },
        ],
    },
    "return_type": "entry",
}

DATA_GRAPHQL_QUERY = """
query SkinScoutRcsbHoloSnapshot($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    rcsb_accession_info {
      initial_release_date
    }
    exptl {
      method
    }
    rcsb_entry_info {
      resolution_combined
    }
    rcsb_primary_citation {
      pdbx_database_id_DOI
      pdbx_database_id_PubMed
      title
      year
    }
    polymer_entities {
      rcsb_id
      rcsb_entity_source_organism {
        ncbi_taxonomy_id
        scientific_name
        taxonomy_lineage {
          id
          name
        }
      }
      rcsb_polymer_entity_container_identifiers {
        asym_ids
        auth_asym_ids
        uniprot_ids
        reference_sequence_identifiers {
          database_accession
          database_name
          provenance_source
        }
      }
      polymer_entity_instances {
        rcsb_id
        rcsb_polymer_entity_instance_container_identifiers {
          asym_id
          auth_asym_id
        }
      }
    }
    nonpolymer_entities {
      rcsb_id
      rcsb_nonpolymer_entity_container_identifiers {
        auth_asym_ids
        asym_ids
        entity_id
        nonpolymer_comp_id
      }
      nonpolymer_comp {
        chem_comp {
          id
          name
          formula
          formula_weight
          type
        }
        rcsb_chem_comp_descriptor {
          SMILES
          InChI
          InChIKey
          SMILES_stereo
        }
      }
      rcsb_nonpolymer_entity_annotation {
        annotation_id
        type
        name
        description
        provenance_source
      }
      nonpolymer_entity_instances {
        rcsb_id
        rcsb_nonpolymer_entity_instance_container_identifiers {
          asym_id
          auth_asym_id
          auth_seq_id
          comp_id
          entity_id
        }
        rcsb_nonpolymer_instance_annotation {
          annotation_id
          type
          name
          description
          provenance_source
        }
        rcsb_target_neighbors {
          atom_id
          comp_id
          distance
          target_asym_id
          target_atom_id
          target_auth_seq_id
          target_comp_id
          target_entity_id
          target_is_bound
          target_seq_id
        }
        rcsb_nonpolymer_struct_conn {
          connect_partner {
            label_alt_id
            label_asym_id
            label_atom_id
            label_comp_id
            label_seq_id
            symmetry
          }
          connect_target {
            auth_asym_id
            auth_seq_id
            label_alt_id
            label_asym_id
            label_atom_id
            label_comp_id
            label_seq_id
            symmetry
          }
          connect_type
          description
          dist_value
          id
          role
          value_order
        }
      }
    }
  }
}
""".strip()


class MirrorError(RuntimeError):
    """Raised when the RCSB snapshot cannot be mirrored completely."""


def _log(message: str) -> None:
    print(f"[rcsb.holo.mirror] {message}", file=sys.stderr, flush=True)


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tmp_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def _remove_outputs(*paths: Path) -> None:
    for path in paths:
        path.unlink(missing_ok=True)
        _tmp_path(path).unlink(missing_ok=True)


def _write_json_atomic(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    tmp.unlink(missing_ok=True)
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_jsonl_gz_atomic(records: list[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    tmp.unlink(missing_ok=True)
    try:
        with tmp.open("wb") as raw_handle:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_handle,
                mtime=0,
            ) as gzip_handle:
                for record in records:
                    line = _canonical_json(record) + "\n"
                    gzip_handle.write(line.encode("utf-8"))
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _post_json(
    *,
    session: requests.Session,
    url: str,
    payload: Mapping[str, Any],
    timeout_s: float,
) -> Mapping[str, Any]:
    response = session.post(url, json=payload, timeout=timeout_s)
    status_code = int(getattr(response, "status_code", 200))
    if status_code >= 500 or status_code in {408, 429}:
        text = str(getattr(response, "text", ""))[:500]
        raise requests.HTTPError(f"transient HTTP {status_code}: {text}")
    if status_code >= 400:
        text = str(getattr(response, "text", ""))[:500]
        raise MirrorError(f"RCSB API request failed: HTTP {status_code} {text}")
    try:
        data = response.json()
    except ValueError as exc:
        raise MirrorError("RCSB API returned non-JSON response") from exc
    if not isinstance(data, Mapping):
        raise MirrorError("RCSB API returned non-object response")
    return data


def _post_json_with_retries(
    *,
    session: requests.Session,
    url: str,
    payload: Mapping[str, Any],
    timeout_s: float,
    max_retries: int,
    retry_backoff_s: float,
) -> Mapping[str, Any]:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return _post_json(
                session=session,
                url=url,
                payload=payload,
                timeout_s=timeout_s,
            )
        except (requests.RequestException, requests.HTTPError) as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            _log(f"retry RCSB request after transient error ({attempt + 1}/{max_retries}): {exc}")
            if retry_backoff_s:
                time.sleep(retry_backoff_s * (attempt + 1))
    raise MirrorError(
        f"RCSB API request failed after {max_retries + 1} attempts: {last_error}"
    ) from last_error


def _search_payload(
    cutoff: str,
    *,
    min_nonpolymer_mw: float = DEFAULT_MIN_NONPOLYMER_MW,
    start: int,
    rows: int,
) -> dict[str, Any]:
    payload = json.loads(json.dumps(SEARCH_QUERY_CONTRACT))
    payload["query"]["nodes"][1]["parameters"]["value"] = cutoff
    payload["query"]["nodes"][3]["parameters"]["value"] = min_nonpolymer_mw
    payload["request_options"] = {
        "paginate": {"start": start, "rows": rows},
        "results_content_type": ["experimental"],
    }
    return payload


def _extract_search_page(payload: Mapping[str, Any], *, start: int) -> tuple[int, list[str]]:
    total = payload.get("total_count")
    result_set = payload.get("result_set")
    if not isinstance(total, int) or total < 0:
        raise MirrorError(f"RCSB Search API missing valid total_count at start={start}")
    if not isinstance(result_set, list):
        raise MirrorError(f"RCSB Search API missing result_set list at start={start}")
    identifiers: list[str] = []
    for item in result_set:
        if not isinstance(item, Mapping) or not isinstance(item.get("identifier"), str):
            raise MirrorError(f"RCSB Search API malformed result at start={start}")
        entry_id = item["identifier"].strip().upper()
        if not entry_id:
            raise MirrorError(f"RCSB Search API returned blank identifier at start={start}")
        identifiers.append(entry_id)
    return total, identifiers


def _enumerate_entry_ids(
    *,
    session: requests.Session,
    search_api_url: str,
    cutoff: str,
    min_nonpolymer_mw: float,
    page_size: int,
    timeout_s: float,
    max_retries: int,
    retry_backoff_s: float,
) -> list[str]:
    ids: list[str] = []
    expected_total: int | None = None
    start = 0
    while True:
        page_payload = _search_payload(
            cutoff,
            min_nonpolymer_mw=min_nonpolymer_mw,
            start=start,
            rows=page_size,
        )
        page = _post_json_with_retries(
            session=session,
            url=search_api_url,
            payload=page_payload,
            timeout_s=timeout_s,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
        )
        total, page_ids = _extract_search_page(page, start=start)
        if expected_total is None:
            expected_total = total
            if expected_total == 0:
                raise MirrorError("RCSB Search API returned zero candidate entries")
        elif total != expected_total:
            raise MirrorError(
                f"RCSB Search API total_count changed: {expected_total} -> {total}"
            )
        overlap = set(ids).intersection(page_ids)
        if overlap:
            raise MirrorError(
                "RCSB Search API returned duplicate entry IDs across pages: "
                f"{sorted(overlap)}"
            )
        if len(page_ids) != len(set(page_ids)):
            raise MirrorError(f"RCSB Search API returned duplicate entry IDs on page {start}")
        ids.extend(page_ids)
        _log(f"search page start={start}: ids={len(ids)}/{expected_total}")
        if len(ids) >= expected_total:
            break
        if not page_ids:
            raise MirrorError(
                f"RCSB Search API pagination stopped early at {len(ids)}/{expected_total}"
            )
        start += page_size
    if len(ids) != expected_total:
        raise MirrorError(
            f"RCSB Search API completeness mismatch: {len(ids)} != {expected_total}"
        )
    sorted_ids = sorted(ids)
    if ids != sorted_ids:
        _log("RCSB Search API order was not sorted; canonicalizing output order")
    return sorted_ids


def _fetch_entries_batch(
    *,
    session: requests.Session,
    data_api_url: str,
    ids: list[str],
    timeout_s: float,
    max_retries: int,
    retry_backoff_s: float,
) -> list[Mapping[str, Any]]:
    payload = {
        "query": DATA_GRAPHQL_QUERY,
        "variables": {"ids": ids},
    }
    response = _post_json_with_retries(
        session=session,
        url=data_api_url,
        payload=payload,
        timeout_s=timeout_s,
        max_retries=max_retries,
        retry_backoff_s=retry_backoff_s,
    )
    if response.get("errors"):
        raise MirrorError(f"RCSB Data API GraphQL errors: {response['errors']}")
    data = response.get("data")
    if not isinstance(data, Mapping):
        raise MirrorError("RCSB Data API response missing data object")
    entries = data.get("entries")
    if not isinstance(entries, list):
        raise MirrorError("RCSB Data API response missing entries list")
    observed: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise MirrorError("RCSB Data API returned malformed entry")
        entry_id = entry.get("rcsb_id")
        if not isinstance(entry_id, str) or not entry_id.strip():
            raise MirrorError("RCSB Data API entry missing rcsb_id")
        normalized_id = entry_id.strip().upper()
        if normalized_id in observed:
            raise MirrorError(f"RCSB Data API returned duplicate entry {normalized_id}")
        observed[normalized_id] = entry
    requested = set(ids)
    if set(observed) != requested:
        missing = sorted(requested - set(observed))
        extra = sorted(set(observed) - requested)
        raise MirrorError(
            f"RCSB Data API batch completeness mismatch: missing={missing}, extra={extra}"
        )
    return [observed[entry_id] for entry_id in ids]


def _batches(values: list[str], batch_size: int) -> Iterable[list[str]]:
    for offset in range(0, len(values), batch_size):
        yield values[offset : offset + batch_size]


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _has_human_lineage(entity: Mapping[str, Any]) -> bool:
    organisms = _list(entity.get("rcsb_entity_source_organism"))
    for organism in organisms:
        org = _mapping(organism)
        if str(org.get("ncbi_taxonomy_id") or "") == "9606":
            return True
        for lineage in _list(org.get("taxonomy_lineage")):
            if str(_mapping(lineage).get("id") or "") == "9606":
                return True
    return False


def _human_uniprot_asym_mappings(entry: Mapping[str, Any]) -> list[dict[str, str]]:
    mappings: list[dict[str, str]] = []
    for entity in _list(entry.get("polymer_entities")):
        entity_map = _mapping(entity)
        if not _has_human_lineage(entity_map):
            continue
        identifiers = _mapping(entity_map.get("rcsb_polymer_entity_container_identifiers"))
        references = [
            _mapping(value)
            for value in _list(identifiers.get("reference_sequence_identifiers"))
            if str(_mapping(value).get("database_name") or "").strip().casefold()
            == "uniprot"
            and str(_mapping(value).get("database_accession") or "").strip()
        ]
        references = sorted(
            {
                (
                    str(reference.get("database_accession") or "").strip(),
                    str(reference.get("provenance_source") or "").strip(),
                )
                for reference in references
            }
        )
        asym_ids = [str(value) for value in _list(identifiers.get("asym_ids")) if value]
        auth_asym_ids = [str(value) for value in _list(identifiers.get("auth_asym_ids")) if value]
        instances = _list(entity_map.get("polymer_entity_instances"))
        if instances:
            for instance in instances:
                instance_map = _mapping(instance)
                inst_ids = _mapping(
                    instance_map.get(
                        "rcsb_polymer_entity_instance_container_identifiers"
                    )
                )
                asym = str(inst_ids.get("asym_id") or "")
                auth_asym = str(inst_ids.get("auth_asym_id") or "")
                for uniprot_id, provenance_source in references:
                    mappings.append(
                        {
                            "entity_id": str(entity_map.get("rcsb_id") or ""),
                            "instance_id": str(instance_map.get("rcsb_id") or ""),
                            "asym_id": asym,
                            "auth_asym_id": auth_asym,
                            "uniprot_id": uniprot_id,
                            "reference_database": "UniProt",
                            "reference_provenance_source": provenance_source,
                        }
                    )
        else:
            for uniprot_id, provenance_source in references:
                for asym in asym_ids or [""]:
                    mappings.append(
                        {
                            "entity_id": str(entity_map.get("rcsb_id") or ""),
                            "instance_id": "",
                            "asym_id": asym,
                            "auth_asym_id": auth_asym_ids[0] if auth_asym_ids else "",
                            "uniprot_id": uniprot_id,
                            "reference_database": "UniProt",
                            "reference_provenance_source": provenance_source,
                        }
                    )
    return sorted(mappings, key=lambda row: tuple(row.values()))


def _subject_annotations(entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    for entity in _list(entry.get("nonpolymer_entities")):
        entity_map = _mapping(entity)
        for annotation in _list(entity_map.get("rcsb_nonpolymer_entity_annotation")):
            annotation_map = _mapping(annotation)
            if (
                str(annotation_map.get("type") or "").strip().upper()
                == "SUBJECT_OF_INVESTIGATION"
                and str(annotation_map.get("provenance_source") or "").strip().upper()
                == "PDB"
            ):
                annotations.append(
                    {
                        "scope": "nonpolymer_entity",
                        "rcsb_id": entity_map.get("rcsb_id"),
                        "annotation": dict(annotation_map),
                    }
                )
        for instance in _list(entity_map.get("nonpolymer_entity_instances")):
            instance_map = _mapping(instance)
            for annotation in _list(
                instance_map.get("rcsb_nonpolymer_instance_annotation")
            ):
                annotation_map = _mapping(annotation)
                if (
                    str(annotation_map.get("type") or "").strip().upper()
                    == "SUBJECT_OF_INVESTIGATION"
                    and str(annotation_map.get("provenance_source") or "").strip().upper()
                    == "PDB"
                ):
                    annotations.append(
                        {
                            "scope": "nonpolymer_instance",
                            "rcsb_id": instance_map.get("rcsb_id"),
                            "annotation": dict(annotation_map),
                        }
                    )
    return sorted(annotations, key=lambda item: _canonical_json(item))


def _nonpolymer_entity_instances(entry: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    instances: list[Mapping[str, Any]] = []
    for entity in _list(entry.get("nonpolymer_entities")):
        instances.extend(
            _mapping(instance)
            for instance in _list(_mapping(entity).get("nonpolymer_entity_instances"))
        )
    return instances


def _nonpolymer_struct_conn(entry: Mapping[str, Any]) -> list[Any]:
    contacts: list[Any] = []
    for instance in _nonpolymer_entity_instances(entry):
        contacts.extend(_list(instance.get("rcsb_nonpolymer_struct_conn")))
    return contacts


def _parse_iso_date(value: object, label: str) -> date:
    text = str(value or "").strip()[:10]
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise MirrorError(f"invalid {label}: {value!r}") from exc


def _normalize_record(
    entry: Mapping[str, Any],
    *,
    release_cutoff: date,
) -> dict[str, Any]:
    required_keys = {
        "rcsb_id",
        "rcsb_accession_info",
        "exptl",
        "rcsb_entry_info",
        "polymer_entities",
        "nonpolymer_entities",
    }
    missing = sorted(key for key in required_keys if key not in entry)
    if missing:
        raise MirrorError(f"RCSB Data API entry {entry.get('rcsb_id')} missing keys {missing}")
    entry_id = str(entry["rcsb_id"]).upper()
    accession = _mapping(entry.get("rcsb_accession_info"))
    release_date = accession.get("initial_release_date")
    if not isinstance(release_date, str) or not release_date:
        raise MirrorError(f"RCSB entry {entry_id} missing initial_release_date")
    parsed_release_date = _parse_iso_date(
        release_date,
        f"RCSB entry {entry_id} initial_release_date",
    )
    if parsed_release_date < release_cutoff:
        raise MirrorError(
            f"RCSB entry {entry_id} predates release cutoff: "
            f"{parsed_release_date.isoformat()} < {release_cutoff.isoformat()}"
        )
    experiments = _list(entry.get("exptl"))
    if not experiments:
        raise MirrorError(f"RCSB entry {entry_id} has no experimental methods")
    mappings = _human_uniprot_asym_mappings(entry)
    annotations = _subject_annotations(entry)
    if not annotations:
        raise MirrorError(f"RCSB entry {entry_id} has no SUBJECT_OF_INVESTIGATION annotation")
    nonpolymer_instances = _nonpolymer_entity_instances(entry)
    return {
        "schema_version": SCHEMA_VERSION,
        "entry_id": entry_id,
        "initial_release_date": release_date,
        "experimental_methods": experiments,
        "resolution_combined": _mapping(entry.get("rcsb_entry_info")).get(
            "resolution_combined"
        ),
        "primary_citation": entry.get("rcsb_primary_citation"),
        "human_polymer_uniprot_asym_mappings": mappings,
        "nonpolymer_entities": entry.get("nonpolymer_entities"),
        "nonpolymer_entity_instances": nonpolymer_instances,
        "subject_of_investigation_annotations": annotations,
        "rcsb_target_neighbors": [
            {
                "instance_id": _mapping(instance).get("rcsb_id"),
                "neighbors": _list(_mapping(instance).get("rcsb_target_neighbors")),
            }
            for instance in nonpolymer_instances
        ],
        "rcsb_nonpolymer_struct_conn": _nonpolymer_struct_conn(entry),
    }


def mirror_rcsb_holo_snapshot(
    *,
    raw_jsonl_gz: Path,
    manifest: Path,
    release_cutoff: str = DEFAULT_RELEASE_CUTOFF,
    min_nonpolymer_mw: float = DEFAULT_MIN_NONPOLYMER_MW,
    batch_size: int = 100,
    timeout_s: float = 60.0,
    max_retries: int = 5,
    retry_backoff_s: float = 2.0,
    search_api_url: str = SEARCH_API_URL,
    data_api_url: str = DATA_API_GRAPHQL_URL,
    session: requests.Session | None = None,
    retrieved_at: datetime | None = None,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise MirrorError("--batch-size must be positive")
    if not math.isfinite(min_nonpolymer_mw) or min_nonpolymer_mw < 0:
        raise MirrorError("--min-nonpolymer-mw must be finite and non-negative")
    if timeout_s <= 0:
        raise MirrorError("--timeout-s must be positive")
    if max_retries < 0:
        raise MirrorError("--max-retries must be non-negative")
    if retry_backoff_s < 0:
        raise MirrorError("--retry-backoff-s must be non-negative")
    parsed_release_cutoff = _parse_iso_date(release_cutoff, "--release-cutoff")
    _remove_outputs(raw_jsonl_gz, manifest)
    client = session or requests.Session()
    try:
        entry_ids = _enumerate_entry_ids(
            session=client,
            search_api_url=search_api_url,
            cutoff=release_cutoff,
            min_nonpolymer_mw=min_nonpolymer_mw,
            page_size=batch_size,
            timeout_s=timeout_s,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
        )
        records: list[dict[str, Any]] = []
        for batch_number, batch in enumerate(_batches(entry_ids, batch_size), start=1):
            entries = _fetch_entries_batch(
                session=client,
                data_api_url=data_api_url,
                ids=batch,
                timeout_s=timeout_s,
                max_retries=max_retries,
                retry_backoff_s=retry_backoff_s,
            )
            records.extend(
                _normalize_record(entry, release_cutoff=parsed_release_cutoff)
                for entry in entries
            )
            _log(
                f"graphql batch {batch_number}: records={len(records)}/{len(entry_ids)}"
            )
        records.sort(key=lambda record: str(record["entry_id"]))
        if [record["entry_id"] for record in records] != entry_ids:
            raise MirrorError("RCSB Data API record order/completeness mismatch")
        _write_jsonl_gz_atomic(records, raw_jsonl_gz)
        raw_bytes = raw_jsonl_gz.stat().st_size
        raw_sha256 = _sha256_file(raw_jsonl_gz)
        search_contract = _search_payload(
            release_cutoff,
            min_nonpolymer_mw=min_nonpolymer_mw,
            start=0,
            rows=batch_size,
        )
        retrieval_time = retrieved_at or datetime.now(UTC)
        records_with_mapping = sum(
            bool(record["human_polymer_uniprot_asym_mappings"])
            for record in records
        )
        records_with_citation = sum(
            bool(_mapping(record.get("primary_citation")))
            for record in records
        )
        manifest_payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "source": SOURCE_NAME,
            "source_urls": {
                "search_api": search_api_url,
                "data_api_graphql": data_api_url,
            },
            "source_license": {
                "name": "CC0",
                "url": RCSB_CC0_LICENSE_URL,
            },
            "source_version": {
                "archive_release_model": "continuous",
                "snapshot_retrieved_at": retrieval_time.astimezone(UTC).isoformat(),
                "entry_initial_release_date_gte": release_cutoff,
            },
            "api_versions": {
                "search_api": "v2",
                "data_api": "GraphQL live schema; exact query hash pinned",
            },
            "usage_contract": {
                "positive_only_direct_contact_evaluation_source": True,
                "never_training_or_calibration": True,
                "forbidden_external_mappings": ["DrugBank", "ChEMBL"],
            },
            "api_query_contract": {
                "search": search_contract,
                "graphql": DATA_GRAPHQL_QUERY,
            },
            "query_sha256s": {
                "search": _sha256_text(_canonical_json(search_contract)),
                "graphql": _sha256_text(DATA_GRAPHQL_QUERY),
            },
            "release_cutoff": release_cutoff,
            "retrieved_at": retrieval_time.astimezone(UTC).isoformat(),
            "batch_size": batch_size,
            "timeout_s": timeout_s,
            "max_retries": max_retries,
            "candidate_count": len(entry_ids),
            "candidate_record_counts": {
                "with_human_uniprot_asym_mapping": records_with_mapping,
                "without_human_uniprot_asym_mapping": len(records) - records_with_mapping,
                "with_primary_citation": records_with_citation,
                "without_primary_citation": len(records) - records_with_citation,
            },
            "filters": {
                "experimental_entries": True,
                "human_taxonomy_lineage_id": 9606,
                "initial_release_date_gte": release_cutoff,
                "nonpolymer_molecular_weight_gt": min_nonpolymer_mw,
                "pdb_native_subject_of_investigation": True,
            },
            "raw_jsonl_gz": {
                "path": str(raw_jsonl_gz.resolve()),
                "sha256": raw_sha256,
                "bytes": raw_bytes,
                "rows": len(records),
            },
        }
        _write_json_atomic(manifest_payload, manifest)
        _log(f"mirror complete: rows={len(records)}, output={raw_jsonl_gz}")
        return manifest_payload
    except Exception:
        _remove_outputs(raw_jsonl_gz, manifest)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-cutoff", default=DEFAULT_RELEASE_CUTOFF)
    parser.add_argument(
        "--min-nonpolymer-mw",
        type=float,
        default=DEFAULT_MIN_NONPOLYMER_MW,
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-backoff-s", type=float, default=2.0)
    parser.add_argument("--raw-jsonl-gz", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--search-api-url", default=SEARCH_API_URL)
    parser.add_argument("--data-api-url", default=DATA_API_GRAPHQL_URL)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = mirror_rcsb_holo_snapshot(
            raw_jsonl_gz=args.raw_jsonl_gz,
            manifest=args.manifest,
            release_cutoff=args.release_cutoff,
            min_nonpolymer_mw=args.min_nonpolymer_mw,
            batch_size=args.batch_size,
            timeout_s=args.timeout_s,
            max_retries=args.max_retries,
            retry_backoff_s=args.retry_backoff_s,
            search_api_url=args.search_api_url,
            data_api_url=args.data_api_url,
        )
    except MirrorError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        "Mirrored RCSB holo snapshot: "
        f"{manifest['raw_jsonl_gz']['rows']} records -> "
        f"{manifest['raw_jsonl_gz']['path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
