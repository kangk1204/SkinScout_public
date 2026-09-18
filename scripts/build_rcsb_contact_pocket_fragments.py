#!/usr/bin/env python3
"""Build provenance-bound RCSB ligand-contact pocket fragments for evaluation."""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import json
import logging
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import requests
from Bio.PDB import MMCIF2Dict, MMCIFParser, PDBIO
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.Structure import Structure


SCHEMA_VERSION = "skinscout.rcsb-contact-pocket-fragments.v1"
PANEL_SCHEMA_VERSION = "skinscout.rcsb-holo-direct-contact-panel.v1"
SOURCE_SCHEMA_VERSION = "skinscout.rcsb-holo-contact-snapshot.v1"
RCSB_CC0_LICENSE_URL = "https://www.rcsb.org/pages/policies"
RCSB_CIF_URL_TEMPLATE = "https://files.rcsb.org/download/{entry}.cif.gz"
CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
LOG = logging.getLogger("rcsb-contact-pocket-fragments")
REQUIRED_PAIR_COLUMNS = {
    "pair_id",
    "entry_id",
    "component_id",
    "instance_id",
    "uniprot",
    "ligand_key",
    "contacted_residue_count",
    "contacted_residues",
    "is_dual_cold",
}
INDEX_FIELDS = [
    "pair_id",
    "entry_id",
    "component_id",
    "instance_id",
    "uniprot",
    "ligand_key",
    "contacted_residue_count",
    "seed_residue_count",
    "fragment_residue_count",
    "context_residues",
    "source_cif_url",
    "source_cif_gz_path",
    "source_cif_gz_sha256",
    "fragment_path",
    "fragment_sha256",
    "chain_id_map_json",
    "contacted_residues_json",
    "extracted_contacted_residues_json",
]
EXCLUSION_FIELDS = [
    "pair_id",
    "entry_id",
    "component_id",
    "instance_id",
    "uniprot",
    "ligand_key",
    "reason",
    "detail",
]


class FragmentError(RuntimeError):
    """Raised when provenance, retrieval, or extraction fails."""


def _clean(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.upper() in {"", "NA", "N/A", "NULL", "NAN", "NONE"} else text


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    if root.exists():
        for path in sorted(
            p
            for p in root.rglob("*")
            if p.is_file() and p.name != ".snakemake_timestamp"
        ):
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(_sha256(path).encode("ascii"))
            digest.update(b"\0")
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        raise FragmentError(f"{label} is missing or empty: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FragmentError(f"Unable to parse {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise FragmentError(f"{label} must be a JSON object: {path}")
    return payload


def _resolve_manifest_path(value: object, manifest_path: Path, label: str) -> Path:
    text = _clean(value)
    if not text:
        raise FragmentError(f"{label} path is blank")
    path = Path(text)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _parquet_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        raise FragmentError(f"pairs parquet is missing or empty: {path}")
    try:
        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception as exc:
        raise FragmentError(f"Unable to inspect pairs parquet: {path}") from exc


def _csv_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open("rb") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def _as_list(value: object) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        out = value.tolist()
        return out if isinstance(out, list) else [out]
    return [value]


def _mmcif_values(payload: Mapping[str, object], key: str) -> list[str]:
    value = payload.get(key)
    if value is None:
        return []
    values = _as_list(value)
    return [str(item) for item in values]


def _validate_panel_manifest(path: Path, pairs_path: Path) -> dict[str, Any]:
    manifest = _read_json(path, "panel manifest")
    if manifest.get("schema_version") != PANEL_SCHEMA_VERSION:
        raise FragmentError(f"panel manifest schema_version must be {PANEL_SCHEMA_VERSION}")
    contract = manifest.get("contract")
    if not isinstance(contract, Mapping):
        raise FragmentError("panel manifest missing contract object")
    for key in (
        "positive_only",
        "no_inferred_negatives",
        "no_affinities_or_calibration",
        "never_training_or_model_selection",
        "all_ranking_rows_dual_cold",
    ):
        if contract.get(key) is not True:
            raise FragmentError(f"panel manifest contract.{key} must be true")
    outputs = manifest.get("outputs")
    pair_meta = outputs.get("pairs") if isinstance(outputs, Mapping) else None
    if not isinstance(pair_meta, Mapping):
        raise FragmentError("panel manifest missing outputs.pairs")
    expected_path = _resolve_manifest_path(
        pair_meta.get("path"), path, "panel manifest outputs.pairs"
    )
    if expected_path != pairs_path.resolve():
        raise FragmentError("panel manifest outputs.pairs.path does not match --pairs-parquet")
    if pair_meta.get("sha256") != _sha256(pairs_path):
        raise FragmentError("panel manifest outputs.pairs.sha256 does not match --pairs-parquet")
    if int(pair_meta.get("rows", -1)) != _parquet_rows(pairs_path):
        raise FragmentError("panel manifest outputs.pairs.rows does not match --pairs-parquet")
    inputs = manifest.get("inputs")
    source_manifest = inputs.get("source_manifest") if isinstance(inputs, Mapping) else None
    if not isinstance(source_manifest, Mapping) or source_manifest.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise FragmentError("panel manifest must bind the RCSB holo contact source manifest")
    source_path = _resolve_manifest_path(
        source_manifest.get("path"), path, "panel source manifest"
    )
    if source_manifest.get("sha256") != _sha256(source_path):
        raise FragmentError("panel source manifest sha256 is stale")
    source_payload = _read_json(source_path, "RCSB holo contact source manifest")
    if source_payload.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise FragmentError(f"source manifest schema_version must be {SOURCE_SCHEMA_VERSION}")
    source_license = source_payload.get("source_license")
    source_contract = source_payload.get("usage_contract")
    if (
        not isinstance(source_license, Mapping)
        or source_license.get("url") != RCSB_CC0_LICENSE_URL
        or not isinstance(source_contract, Mapping)
        or source_contract.get("positive_only_direct_contact_evaluation_source") is not True
        or source_contract.get("never_training_or_calibration") is not True
    ):
        raise FragmentError("RCSB source provenance or usage contract is invalid")
    return manifest


def _read_pairs(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    missing = sorted(REQUIRED_PAIR_COLUMNS - set(frame.columns))
    if missing:
        raise FragmentError(f"pairs parquet missing required column(s): {', '.join(missing)}")
    if frame["pair_id"].astype(str).duplicated().any():
        raise FragmentError("pairs parquet contains duplicate pair_id values")
    valid_flags = frame["is_dual_cold"].map(
        lambda value: isinstance(value, (bool, np.bool_))
    )
    if not bool(valid_flags.all()):
        raise FragmentError("pairs parquet contains non-boolean is_dual_cold values")
    strict = frame[frame["is_dual_cold"].map(bool)].copy()
    if strict.empty:
        raise FragmentError("pairs parquet contains no strict dual-cold pairs")
    return strict.sort_values(
        ["entry_id", "component_id", "instance_id", "pair_id"], kind="mergesort"
    ).reset_index(drop=True)


def _parse_contacted_residues(value: object) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError) as exc:
                raise FragmentError("contacted_residues string is not a parseable list") from exc
            tokens = [str(item).strip() for item in _as_list(parsed)]
        else:
            tokens = [part.strip() for part in re.split(r"[;,]", text)]
    else:
        tokens = [str(item).strip() for item in _as_list(value)]
    tokens = [token for token in tokens if token]
    bad = [token for token in tokens if len(token.split("|")) != 4]
    if bad:
        raise FragmentError(f"contacted_residues token(s) must have four pipe fields: {bad[:3]}")
    return sorted(set(tokens))


def _safe_name(value: object) -> str:
    text = _clean(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("._") or "blank"


def _download_cif_gz(entry: str, path: Path, *, retries: int, timeout: float) -> dict[str, Any]:
    url = RCSB_CIF_URL_TEMPLATE.format(entry=entry.upper())
    session = requests.Session()
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            data = response.content
            if not data:
                raise FragmentError(f"empty RCSB response for {entry}")
            gzip.decompress(data)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return {
                "entry_id": entry.upper(),
                "url": url,
                "path": str(path),
                "sha256": _sha256_bytes(data),
                "bytes": len(data),
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }
        except (OSError, requests.RequestException, FragmentError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2.0, 0.25 * attempt))
    raise FragmentError(f"failed to download verified RCSB mmCIF gzip for {entry}: {last_error}")


def _load_structure(entry: str, gzip_path: Path) -> tuple[Any, dict[str, set[tuple[str, str, str]]]]:
    try:
        text = gzip.decompress(gzip_path.read_bytes()).decode("utf-8")
    except Exception as exc:
        raise FragmentError(f"unable to decompress RCSB mmCIF gzip for {entry}") from exc
    parser = MMCIFParser(QUIET=True, auth_chains=False, auth_residues=False)
    try:
        structure = parser.get_structure(entry, StringIO(text))
    except Exception as exc:
        raise FragmentError(f"unable to parse RCSB mmCIF for {entry}") from exc
    try:
        mmcif = MMCIF2Dict.MMCIF2Dict(StringIO(text))
    except Exception as exc:
        raise FragmentError(f"unable to index RCSB mmCIF atom_site rows for {entry}") from exc
    labels = _mmcif_values(mmcif, "_atom_site.label_asym_id")
    auth_seq = _mmcif_values(mmcif, "_atom_site.auth_seq_id")
    label_seq = _mmcif_values(mmcif, "_atom_site.label_seq_id")
    comp = _mmcif_values(mmcif, "_atom_site.label_comp_id")
    if not (labels and len({len(labels), len(auth_seq), len(label_seq), len(comp)}) == 1):
        raise FragmentError(f"RCSB mmCIF atom_site fields are incomplete for {entry}")
    contact_lookup: dict[str, set[tuple[str, str, str]]] = {}
    for chain_id, auth, seq, resname in zip(labels, auth_seq, label_seq, comp, strict=True):
        if not chain_id or not seq or seq in {".", "?"}:
            continue
        token = f"{chain_id}|{auth}|{seq}|{resname}"
        contact_lookup.setdefault(token, set()).add((chain_id, seq, resname))
    return structure, contact_lookup


def _eligible_by_chain(structure: Any) -> dict[str, list[Any]]:
    model = next(structure.get_models(), None)
    if model is None:
        raise FragmentError("RCSB mmCIF contains no models")
    by_chain: dict[str, list[Any]] = {}
    for chain in model:
        residues = [
            residue
            for residue in chain
            if residue.id[0] == " " and is_aa(residue, standard=True)
        ]
        if residues:
            by_chain[chain.id] = residues
    return by_chain


def _select_residues(
    structure: Any,
    contact_lookup: Mapping[str, set[tuple[str, str, str]]],
    contact_tokens: Iterable[str],
    context_residues: int,
) -> tuple[list[Any], list[str]]:
    by_chain = _eligible_by_chain(structure)
    selected: dict[tuple[str, tuple[Any, ...]], Any] = {}
    extracted_tokens: set[str] = set()
    for token in contact_tokens:
        matches = sorted(contact_lookup.get(token, set()))
        for chain_id, label_seq, comp_id in matches:
            residues = by_chain.get(chain_id, [])
            positions = {
                str(residue.id[1]): index
                for index, residue in enumerate(residues)
                if residue.get_resname() == comp_id
            }
            if label_seq not in positions:
                continue
            center = positions[label_seq]
            start = max(0, center - context_residues)
            stop = min(len(residues), center + context_residues + 1)
            for residue in residues[start:stop]:
                selected[(chain_id, residue.id)] = residue
            extracted_tokens.add(token)
    ordered = [
        residue
        for _key, residue in sorted(
            selected.items(),
            key=lambda item: (item[0][0], item[0][1][1], str(item[0][1][2])),
        )
    ]
    return ordered, sorted(extracted_tokens)


def _safe_chain_map(residues: Iterable[Any]) -> dict[str, str]:
    chains = sorted({residue.get_parent().id for residue in residues})
    if len(chains) > len(CHAIN_IDS):
        raise FragmentError("too many source chains to remap into one-character PDB chain IDs")
    return {chain: CHAIN_IDS[index] for index, chain in enumerate(chains)}


def _write_fragment(residues: list[Any], path: Path) -> dict[str, str]:
    chain_map = _safe_chain_map(residues)
    structure = Structure(path.stem)
    model = Model(0)
    structure.add(model)
    chains: dict[str, Chain] = {}
    for residue in residues:
        source_chain = residue.get_parent().id
        mapped = chain_map[source_chain]
        if mapped not in chains:
            chains[mapped] = Chain(mapped)
            model.add(chains[mapped])
        clone = deepcopy(residue)
        clone.detach_parent()
        chains[mapped].add(clone)
    path.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(path))
    return chain_map


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clean_outputs(paths: Iterable[Path], dirs: Iterable[Path]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)
        path.with_name(f".{path.name}.tmp").unlink(missing_ok=True)
    for path in dirs:
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def _pair_exclusion(row: Any, reason: str, detail: str) -> dict[str, str]:
    return {
        "pair_id": _clean(row.pair_id),
        "entry_id": _clean(row.entry_id).upper(),
        "component_id": _clean(row.component_id),
        "instance_id": _clean(row.instance_id),
        "uniprot": _clean(row.uniprot),
        "ligand_key": _clean(row.ligand_key),
        "reason": reason,
        "detail": detail,
    }


def _extract_entry_fragments(
    task: tuple[
        str,
        list[dict[str, object]],
        Path,
        Path,
        int,
        str,
    ],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    entry, pair_rows, coordinate_path, fragment_dir, context_residues, cif_sha256 = task
    structure, contact_lookup = _load_structure(entry, coordinate_path)
    index_rows: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    for record in pair_rows:
        row = argparse.Namespace(**record)
        contact_tokens = _parse_contacted_residues(row.contacted_residues)
        try:
            recorded_contact_count = int(row.contacted_residue_count)
        except (TypeError, ValueError) as exc:
            raise FragmentError(
                f"pair {row.pair_id} has invalid contacted_residue_count"
            ) from exc
        if recorded_contact_count != len(contact_tokens):
            raise FragmentError(
                f"pair {row.pair_id} contacted_residue_count does not match "
                "contacted_residues"
            )
        if not contact_tokens:
            exclusions.append(
                _pair_exclusion(
                    row,
                    "no_contacted_residue_seeds",
                    "blank contacted_residues",
                )
            )
            continue
        residues, extracted = _select_residues(
            structure,
            contact_lookup,
            contact_tokens,
            context_residues,
        )
        if not extracted:
            exclusions.append(
                _pair_exclusion(
                    row,
                    "contact_residues_not_extractable",
                    json.dumps(contact_tokens, sort_keys=True),
                )
            )
            continue
        fragment_name = (
            f"{entry}_{_safe_name(row.component_id)}_{_safe_name(row.instance_id)}_"
            f"{_safe_name(row.pair_id)[:16]}_context{context_residues}.pdb"
        )
        fragment_path = fragment_dir / fragment_name
        chain_map = _write_fragment(residues, fragment_path)
        index_rows.append(
            {
                "pair_id": _clean(row.pair_id),
                "entry_id": entry,
                "component_id": _clean(row.component_id),
                "instance_id": _clean(row.instance_id),
                "uniprot": _clean(row.uniprot),
                "ligand_key": _clean(row.ligand_key),
                "contacted_residue_count": recorded_contact_count,
                "seed_residue_count": len(contact_tokens),
                "fragment_residue_count": len(residues),
                "context_residues": context_residues,
                "source_cif_url": RCSB_CIF_URL_TEMPLATE.format(entry=entry),
                "source_cif_gz_path": f"{entry}.cif.gz",
                "source_cif_gz_sha256": cif_sha256,
                "fragment_path": fragment_name,
                "fragment_sha256": _sha256(fragment_path),
                "chain_id_map_json": json.dumps(
                    chain_map, sort_keys=True, separators=(",", ":")
                ),
                "contacted_residues_json": json.dumps(
                    contact_tokens, sort_keys=True, separators=(",", ":")
                ),
                "extracted_contacted_residues_json": json.dumps(
                    extracted, sort_keys=True, separators=(",", ":")
                ),
            }
        )
    return index_rows, exclusions


def build_fragments(args: argparse.Namespace) -> dict[str, Any]:
    if args.context_residues < 0:
        raise FragmentError("--context-residues must be >= 0")
    if args.retries < 1:
        raise FragmentError("--retries must be >= 1")
    if args.timeout <= 0:
        raise FragmentError("--timeout must be > 0")
    workers = int(getattr(args, "workers", 1))
    if workers < 1:
        raise FragmentError("--workers must be >= 1")

    outputs = [args.out_index, args.out_exclusions, args.out_manifest]
    staging_root = args.fragment_dir.parent / f".{args.fragment_dir.name}.rcsb-contact-staging"
    staging_coordinates = staging_root / "coordinates"
    staging_fragments = staging_root / "fragments"
    _clean_outputs(outputs, [args.coordinate_dir, args.fragment_dir, staging_root])

    try:
        panel_manifest = _validate_panel_manifest(args.panel_manifest, args.pairs_parquet)
        strict_pairs = _read_pairs(args.pairs_parquet)
        staging_coordinates.mkdir(parents=True, exist_ok=True)
        staging_fragments.mkdir(parents=True, exist_ok=True)
        retrieval: dict[str, dict[str, Any]] = {}
        for entry in sorted(set(strict_pairs["entry_id"].astype(str).str.upper())):
            retrieval[entry] = _download_cif_gz(
                entry,
                staging_coordinates / f"{entry}.cif.gz",
                retries=args.retries,
                timeout=args.timeout,
            )

        index_rows: list[dict[str, object]] = []
        exclusions: list[dict[str, object]] = []
        grouped_rows: dict[str, list[dict[str, object]]] = {}
        for record in strict_pairs.to_dict("records"):
            entry = _clean(record["entry_id"]).upper()
            grouped_rows.setdefault(entry, []).append(record)
        tasks = [
            (
                entry,
                grouped_rows[entry],
                staging_coordinates / f"{entry}.cif.gz",
                staging_fragments,
                args.context_residues,
                str(retrieval[entry]["sha256"]),
            )
            for entry in sorted(grouped_rows)
        ]
        LOG.info("Extracting %d RCSB entries with %d worker(s)", len(tasks), workers)
        with ExitStack() as stack:
            if workers == 1:
                iterator = map(_extract_entry_fragments, tasks)
            else:
                pool = stack.enter_context(ProcessPoolExecutor(max_workers=workers))
                iterator = pool.map(_extract_entry_fragments, tasks, chunksize=1)
            for done, (entry_rows, entry_exclusions) in enumerate(iterator, start=1):
                index_rows.extend(entry_rows)
                exclusions.extend(entry_exclusions)
                if done == len(tasks) or done % 10 == 0:
                    LOG.info("Extracted RCSB entries: %d/%d", done, len(tasks))

        accounted = len(index_rows) + len(exclusions)
        if accounted != len(strict_pairs):
            raise FragmentError(
                f"strict pair accounting mismatch: strict={len(strict_pairs)} accounted={accounted}"
            )

        _write_csv(staging_root / "index.csv", INDEX_FIELDS, index_rows)
        _write_csv(staging_root / "exclusions.csv", EXCLUSION_FIELDS, exclusions)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "source_license": {"name": "CC0", "url": RCSB_CC0_LICENSE_URL},
            "source_coordinate_url_template": RCSB_CIF_URL_TEMPLATE,
            "contract": {
                "purpose": "evaluation leakage audit only",
                "positive_only": True,
                "no_inferred_negatives": True,
                "never_prediction": True,
                "never_training": True,
                "never_calibration": True,
                "never_model_selection": True,
                "strict_pairs_only": True,
            },
            "parameters": {
                "context_residues": args.context_residues,
                "residue_identifier_policy": "label_asym_id plus auth_seq_id plus label_seq_id plus label_comp_id",
                "parser": "Bio.PDB.MMCIFParser(auth_chains=False, auth_residues=False)",
                "amino_acid_policy": "standard amino-acid residues only",
            },
            "inputs": {
                "pairs_parquet": {
                    "path": str(args.pairs_parquet.resolve()),
                    "sha256": _sha256(args.pairs_parquet),
                    "rows": _parquet_rows(args.pairs_parquet),
                    "strict_dual_cold_rows": len(strict_pairs),
                },
                "panel_manifest": {
                    "path": str(args.panel_manifest.resolve()),
                    "sha256": _sha256(args.panel_manifest),
                    "schema_version": PANEL_SCHEMA_VERSION,
                },
                "panel_source_manifest": panel_manifest["inputs"]["source_manifest"],
            },
            "retrievals": [
                {
                    **retrieval[key],
                    "path": str((args.coordinate_dir / f"{key}.cif.gz").resolve()),
                }
                for key in sorted(retrieval)
            ],
            "counts": {
                "strict_dual_cold_pairs": len(strict_pairs),
                "indexed": len(index_rows),
                "excluded": len(exclusions),
                "unique_entries": len(retrieval),
            },
            "artifacts": {
                "coordinate_dir": {
                    "path": str(args.coordinate_dir.resolve()),
                    "tree_sha256": _tree_digest(staging_coordinates),
                    "files": len(retrieval),
                },
                "fragment_dir": {
                    "path": str(args.fragment_dir.resolve()),
                    "tree_sha256": _tree_digest(staging_fragments),
                    "files": len(index_rows),
                },
                "index_csv": {
                    "path": str(args.out_index.resolve()),
                    "sha256": _sha256(staging_root / "index.csv"),
                    "rows": len(index_rows),
                },
                "exclusions_csv": {
                    "path": str(args.out_exclusions.resolve()),
                    "sha256": _sha256(staging_root / "exclusions.csv"),
                    "rows": _csv_rows(staging_root / "exclusions.csv"),
                },
            },
        }
        _write_json(staging_root / "manifest.json", manifest)

        staging_coordinates.replace(args.coordinate_dir)
        staging_fragments.replace(args.fragment_dir)
        (staging_root / "index.csv").replace(args.out_index)
        (staging_root / "exclusions.csv").replace(args.out_exclusions)
        (staging_root / "manifest.json").replace(args.out_manifest)
        shutil.rmtree(staging_root, ignore_errors=True)
        return manifest
    except BaseException:
        _clean_outputs(outputs, [args.coordinate_dir, args.fragment_dir, staging_root])
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-parquet", required=True, type=Path)
    parser.add_argument("--panel-manifest", required=True, type=Path)
    parser.add_argument("--coordinate-dir", required=True, type=Path)
    parser.add_argument("--fragment-dir", required=True, type=Path)
    parser.add_argument("--out-index", required=True, type=Path)
    parser.add_argument("--out-exclusions", required=True, type=Path)
    parser.add_argument("--out-manifest", required=True, type=Path)
    parser.add_argument("--context-residues", type=int, default=2)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel mmCIF extraction workers (default: 1)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        build_fragments(parse_args(argv))
    except FragmentError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
