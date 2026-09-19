#!/usr/bin/env python3
"""Build a fail-closed positive-only RCSB holo direct-contact ranking panel."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem import Descriptors


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_activity_retrieval_index import (  # noqa: E402
    _standardize_mol,
    _structure_ligand_key,
)
from eval.build_activity_benchmark import _scaffold  # noqa: E402


SCHEMA_VERSION = "skinscout.rcsb-holo-direct-contact-panel.v1"
SOURCE_SCHEMA_VERSION = "skinscout.rcsb-holo-contact-snapshot.v1"
BENCHMARK_SCHEMA_VERSION = "activity_benchmark.v1"
SCREENABLE_CLUSTER_SCHEMA_VERSION = "skinscout.screenable-target-cluster-map.v2"
RCSB_CC0_LICENSE_URL = "https://www.rcsb.org/pages/policies"
DEFAULT_MIN_MW = 50.0
DUAL_COLD_FLAGS = {
    "claimable",
    "absent_pair_from_train",
    "absent_pair_from_prior_splits",
    "absent_publication_from_train",
    "absent_publication_from_prior_splits",
    "absent_scaffold_from_train",
    "absent_scaffold_from_prior_splits",
    "absent_target_cluster_30_from_train",
    "absent_target_cluster_30_from_prior_splits",
    "absent_target_cluster_50_from_train",
    "absent_target_cluster_50_from_prior_splits",
}
PAIR_COLUMNS = [
    "pair_id",
    "entry_id",
    "component_id",
    "instance_id",
    "initial_release_date",
    "uniprot",
    "target_cluster_30",
    "target_cluster_50",
    "source_inchikey",
    "source_identity_smiles",
    "standard_inchikey",
    "connectivity_key",
    "canonical_smiles",
    "ligand_key",
    "standardization_route",
    "identity_route",
    "scaffold_smiles",
    "scaffold_id",
    "contact_distance_max_angstrom",
    "contacted_residue_count",
    "contacted_residues",
    "structure_doi",
    "structure_pmid",
    "source_document_id",
    "publication_key",
    "evidence_date",
    "is_dual_cold",
    *sorted(DUAL_COLD_FLAGS),
]
EXCLUSION_COLUMNS = ["entry_id", "component_id", "instance_id", "reason", "detail"]
RANKING_COLUMNS = [
    "query_id",
    "ligand_key",
    "standard_inchikey",
    "connectivity_key",
    "canonical_smiles",
    "standardization_route",
    "truth_targets",
    "n_truth_targets",
    "split",
    *sorted(DUAL_COLD_FLAGS),
    "source_entry_ids",
    "source_component_ids",
    "source_instance_ids",
    "source_identity_routes",
    "source_publication_keys",
]


class PanelError(RuntimeError):
    """Raised when provenance or raw data is malformed."""


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


def _clean_series(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("").astype(str).str.strip()
    missing = cleaned.str.upper().isin({"", "NA", "N/A", "NULL", "NAN", "NONE"})
    return cleaned.mask(missing, "")


def _benchmark_ligand_identity(frame: pd.DataFrame) -> pd.Series:
    identity = pd.Series("", index=frame.index, dtype="object")
    for column in ("ligand_inchikey", "ligand_id", "ligand_smiles"):
        if column not in frame.columns:
            continue
        values = _clean_series(frame[column])
        identity = identity.mask(identity.eq(""), values)
    return identity.str.upper()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_hash(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _hash_json(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        raise PanelError(f"{label} is missing or empty: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PanelError(f"Unable to parse {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise PanelError(f"{label} must be a JSON object: {path}")
    return payload


def _resolve_path(value: object, manifest_path: Path, label: str) -> Path:
    text = _clean(value)
    if not text:
        raise PanelError(f"{label} missing path")
    path = Path(text)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _parquet_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        raise PanelError(f"Parquet input is missing or empty: {path}")
    try:
        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception as exc:
        raise PanelError(f"Unable to inspect parquet input: {path}") from exc


def _csv_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        raise PanelError(f"CSV input is missing or empty: {path}")
    with path.open("rb") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def _require_columns(df: pd.DataFrame, path: Path, columns: Iterable[str]) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise PanelError(f"{path} missing required column(s): {', '.join(missing)}")


def _manifest_output_sha(manifest: Mapping[str, Any], name: str) -> str:
    hashes = manifest.get("output_sha256")
    if not isinstance(hashes, Mapping):
        raise PanelError("Benchmark manifest missing output_sha256 object")
    value = hashes.get(name)
    if not isinstance(value, str) or len(value) != 64:
        raise PanelError(f"Benchmark manifest missing output_sha256[{name!r}]")
    return value


def _manifest_output_rows(manifest: Mapping[str, Any], split: str, name: str) -> int:
    del name
    splits = manifest.get("splits")
    counts = splits.get("counts") if isinstance(splits, Mapping) else None
    if not isinstance(counts, Mapping) or split not in counts:
        raise PanelError(f"Benchmark manifest missing row count for {split}")
    try:
        rows = int(counts[split])
    except (TypeError, ValueError) as exc:
        raise PanelError(f"Benchmark manifest has invalid row count for {split}") from exc
    if rows < 0:
        raise PanelError(f"Benchmark manifest has negative row count for {split}")
    return rows


def _validate_benchmark_parquet(
    path: Path,
    manifest: Mapping[str, Any],
    *,
    split: str,
    name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    expected_sha = _manifest_output_sha(manifest, name)
    actual_sha = _sha256(path)
    if actual_sha != expected_sha:
        raise PanelError(f"{name} sha256 does not match benchmark manifest")
    actual_rows = _parquet_rows(path)
    expected_rows = _manifest_output_rows(manifest, split, name)
    if actual_rows != expected_rows:
        raise PanelError(f"{name} row count does not match benchmark manifest")
    columns = [
        "uniprot",
        "publication_key",
        "ligand_inchikey",
        "ligand_id",
        "ligand_smiles",
        "scaffold_id",
        "target_cluster_30",
        "target_cluster_50",
    ]
    df = pd.read_parquet(path, columns=[c for c in columns if c in pq.ParquetFile(path).schema_arrow.names])
    _require_columns(
        df,
        path,
        [
            "uniprot",
            "publication_key",
            "scaffold_id",
            "target_cluster_30",
            "target_cluster_50",
        ],
    )
    for column in [
        "uniprot",
        "publication_key",
        "scaffold_id",
        "target_cluster_30",
        "target_cluster_50",
    ]:
        if _clean_series(df[column]).eq("").any():
            raise PanelError(f"{path} contains blank leakage column: {column}")
    ligand_identity = _benchmark_ligand_identity(df)
    if ligand_identity.eq("").any():
        raise PanelError(f"{path} contains blank ligand leakage identity")
    return df, {"path": str(path.resolve()), "sha256": actual_sha, "rows": actual_rows}


def _validate_source_manifest(path: Path, raw_jsonl_gz: Path) -> dict[str, Any]:
    manifest = _read_json(path, "source manifest")
    if manifest.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise PanelError(f"source manifest schema_version must be {SOURCE_SCHEMA_VERSION}")
    license_record = manifest.get("source_license")
    if not isinstance(license_record, Mapping) or license_record.get("url") != RCSB_CC0_LICENSE_URL:
        raise PanelError("source manifest must bind the RCSB CC0 license URL")
    usage = manifest.get("usage_contract")
    if not isinstance(usage, Mapping) or usage.get("positive_only_direct_contact_evaluation_source") is not True or usage.get("never_training_or_calibration") is not True:
        raise PanelError("source manifest usage contract must be positive-only and never-training")
    raw_record = manifest.get("raw_jsonl_gz")
    if not isinstance(raw_record, Mapping):
        raise PanelError("source manifest missing raw_jsonl_gz object")
    recorded_path = _resolve_path(raw_record.get("path"), path, "source raw_jsonl_gz")
    if recorded_path != raw_jsonl_gz.resolve():
        raise PanelError("raw_jsonl_gz path does not match source manifest")
    if raw_record.get("sha256") != _sha256(raw_jsonl_gz):
        raise PanelError("raw_jsonl_gz sha256 does not match source manifest")
    if int(raw_record.get("bytes", -1)) != raw_jsonl_gz.stat().st_size:
        raise PanelError("raw_jsonl_gz byte count does not match source manifest")
    if int(raw_record.get("rows", -1)) < 0:
        raise PanelError("source manifest raw row count is invalid")
    cutoff = _clean(manifest.get("release_cutoff") or manifest.get("source_version", {}).get("entry_initial_release_date_gte"))
    try:
        date.fromisoformat(cutoff)
    except ValueError as exc:
        raise PanelError("source manifest release cutoff is invalid") from exc
    query_hashes = manifest.get("query_sha256s")
    contract = manifest.get("api_query_contract")
    if not isinstance(query_hashes, Mapping) or not isinstance(contract, Mapping):
        raise PanelError("source manifest missing query hashes/contracts")
    if not all(isinstance(query_hashes.get(key), str) and len(query_hashes[key]) == 64 for key in ("search", "graphql")):
        raise PanelError("source manifest query hashes are incomplete")
    if query_hashes["search"] != _hash_json(contract.get("search")):
        raise PanelError("source manifest search query hash does not match its contract")
    graphql = contract.get("graphql")
    if not isinstance(graphql, str) or query_hashes["graphql"] != _hash_text(graphql):
        raise PanelError("source manifest GraphQL query hash does not match its contract")
    if int(manifest.get("candidate_count", -1)) != int(raw_record.get("rows")):
        raise PanelError("source manifest candidate count does not match raw rows")
    counts = manifest.get("candidate_record_counts")
    if not isinstance(counts, Mapping):
        raise PanelError("source manifest missing candidate_record_counts")
    filters = manifest.get("filters")
    if not isinstance(filters, Mapping):
        raise PanelError("source manifest missing filters object")
    if (
        filters.get("experimental_entries") is not True
        or filters.get("human_taxonomy_lineage_id") != 9606
        or filters.get("pdb_native_subject_of_investigation") is not True
        or _clean(filters.get("initial_release_date_gte")) != cutoff
    ):
        raise PanelError("source manifest search filters are invalid")
    try:
        source_min_mw = float(filters["nonpolymer_molecular_weight_gt"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PanelError("source manifest molecular-weight filter is invalid") from exc
    if not math.isfinite(source_min_mw) or source_min_mw < 0:
        raise PanelError("source manifest molecular-weight filter is invalid")
    return manifest


def _validate_clusters(path: Path, manifest_path: Path) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    df = pd.read_csv(path)
    _require_columns(df, path, ["uniprot", "target_cluster_30", "target_cluster_50"])
    manifest = _read_json(manifest_path, "screenable target cluster manifest")
    if manifest.get("schema_version") != SCREENABLE_CLUSTER_SCHEMA_VERSION:
        raise PanelError(f"screenable target cluster manifest schema_version must be {SCREENABLE_CLUSTER_SCHEMA_VERSION}")
    artifact = manifest.get("artifact")
    if not isinstance(artifact, Mapping):
        raise PanelError("screenable target cluster manifest missing artifact object")
    if _resolve_path(artifact.get("path"), manifest_path, "screenable target clusters") != path.resolve():
        raise PanelError("screenable target cluster path does not match its manifest")
    if artifact.get("sha256") != _sha256(path):
        raise PanelError("screenable target cluster CSV sha256 does not match its manifest")
    if int(artifact.get("rows", -1)) != len(df) or _csv_rows(path) != len(df):
        raise PanelError("screenable target cluster CSV row count does not match its manifest")
    production = manifest.get("production_contract") if isinstance(manifest.get("production_contract"), Mapping) else {}
    universe = manifest.get("universe_policy") if isinstance(manifest.get("universe_policy"), Mapping) else {}
    if production.get("passes") is not True:
        raise PanelError("screenable target cluster production contract must pass")
    if universe.get("evaluation_panel_used") is not False or universe.get("known_target_assistance") is not False:
        raise PanelError("screenable target clusters must be evaluation-panel independent")
    mapping: dict[str, dict[str, str]] = {}
    for idx, row in df.iterrows():
        uniprot = _clean(row["uniprot"])
        c30 = _clean(row["target_cluster_30"])
        c50 = _clean(row["target_cluster_50"])
        if not uniprot or not c30 or not c50:
            raise PanelError(f"{path} has blank target cluster assignment at row {idx}")
        if uniprot in mapping:
            raise PanelError(f"{path} has duplicate target cluster assignment for {uniprot}")
        mapping[uniprot] = {"target_cluster_30": c30, "target_cluster_50": c50}
    return mapping, {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "rows": len(df),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "manifest_schema_version": manifest["schema_version"],
        "production_contract": manifest.get("production_contract"),
        "universe_policy": manifest.get("universe_policy"),
    }


def _stream_raw(path: Path, expected_rows: int, cutoff: str) -> Iterable[dict[str, Any]]:
    seen: set[str] = set()
    previous = ""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.endswith("\n"):
                    raise PanelError(f"raw JSONL line {line_number} is not newline-terminated")
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise PanelError(f"raw JSONL line {line_number} is invalid JSON") from exc
                if not isinstance(record, dict):
                    raise PanelError(f"raw JSONL line {line_number} must be an object")
                required = {
                    "schema_version",
                    "entry_id",
                    "initial_release_date",
                    "human_polymer_uniprot_asym_mappings",
                    "nonpolymer_entities",
                    "subject_of_investigation_annotations",
                    "rcsb_target_neighbors",
                    "rcsb_nonpolymer_struct_conn",
                }
                missing = sorted(required - set(record))
                if missing:
                    raise PanelError(f"raw JSONL line {line_number} missing keys: {missing}")
                if record["schema_version"] != SOURCE_SCHEMA_VERSION:
                    raise PanelError(f"raw JSONL line {line_number} has invalid schema_version")
                entry_id = _clean(record["entry_id"]).upper()
                if not entry_id:
                    raise PanelError(f"raw JSONL line {line_number} has blank entry_id")
                if entry_id in seen:
                    raise PanelError(f"raw JSONL duplicate entry_id: {entry_id}")
                if previous and entry_id <= previous:
                    raise PanelError("raw JSONL entry_id order is not strictly increasing")
                previous = entry_id
                seen.add(entry_id)
                try:
                    if date.fromisoformat(_clean(record["initial_release_date"])[:10]) < date.fromisoformat(cutoff):
                        raise PanelError(f"raw entry {entry_id} predates source cutoff")
                except ValueError as exc:
                    raise PanelError(f"raw entry {entry_id} has invalid initial_release_date") from exc
                yield record
    except OSError as exc:
        raise PanelError(f"Unable to read gzipped raw JSONL: {path}") from exc
    if len(seen) != expected_rows:
        raise PanelError(f"raw JSONL row count mismatch: {len(seen)} != {expected_rows}")


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _map(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _publication(record: Mapping[str, Any], entry_id: str) -> tuple[str, str, str, str]:
    citation = _map(record.get("primary_citation"))
    doi = _clean(citation.get("pdbx_database_id_DOI")).lower()
    pmid = "".join(ch for ch in _clean(citation.get("pdbx_database_id_PubMed")) if ch.isdigit())
    if doi:
        publication_key = f"doi:{doi}"
    elif pmid:
        publication_key = f"pmid:{pmid}"
    else:
        publication_key = f"rcsb_entry:{entry_id}"
    return doi, pmid, f"RCSB:{entry_id}", publication_key


def _entity_by_instance(record: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for entity in _list(record.get("nonpolymer_entities")):
        entity_map = _map(entity)
        for instance in _list(entity_map.get("nonpolymer_entity_instances")):
            instance_id = _clean(_map(instance).get("rcsb_id"))
            if instance_id:
                out[instance_id] = entity_map
    return out


def _subject_sets(record: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    entities: set[str] = set()
    instances: set[str] = set()
    for annotation in _list(record.get("subject_of_investigation_annotations")):
        ann = _map(annotation)
        payload = _map(ann.get("annotation"))
        if _clean(payload.get("type")).upper() != "SUBJECT_OF_INVESTIGATION" or _clean(payload.get("provenance_source")).upper() != "PDB":
            continue
        if ann.get("scope") == "nonpolymer_entity":
            entities.add(_clean(ann.get("rcsb_id")))
        elif ann.get("scope") == "nonpolymer_instance":
            instances.add(_clean(ann.get("rcsb_id")))
    return entities, instances


def _target_lookup(record: Mapping[str, Any]) -> dict[str, set[str]]:
    lookup: dict[str, set[str]] = defaultdict(set)
    for mapping in _list(record.get("human_polymer_uniprot_asym_mappings")):
        item = _map(mapping)
        uniprot = _clean(item.get("uniprot_id"))
        if not uniprot:
            continue
        for key in ("asym_id", "auth_asym_id", "entity_id", "instance_id"):
            value = _clean(item.get(key))
            if value:
                lookup[f"{key}:{value}"].add(uniprot)
    return lookup


def _neighbor_targets(neighbor: Mapping[str, Any], lookup: Mapping[str, set[str]]) -> set[str]:
    targets: set[str] = set()
    for key, field in (("asym_id", "target_asym_id"), ("entity_id", "target_entity_id")):
        value = _clean(neighbor.get(field))
        if value:
            targets.update(lookup.get(f"{key}:{value}", set()))
    return targets


def _component_identity(entity: Mapping[str, Any]) -> tuple[str, str, str, str, str, str]:
    identifiers = _map(entity.get("rcsb_nonpolymer_entity_container_identifiers"))
    comp_id = _clean(identifiers.get("nonpolymer_comp_id")) or _clean(entity.get("rcsb_id")).split("_")[-1]
    comp = _map(entity.get("nonpolymer_comp"))
    descriptors = _map(comp.get("rcsb_chem_comp_descriptor"))
    return (
        comp_id,
        _clean(descriptors.get("SMILES_stereo")),
        _clean(descriptors.get("SMILES")),
        _clean(descriptors.get("InChI")),
        _clean(descriptors.get("InChIKey")),
        _clean(_map(comp.get("chem_comp")).get("formula")),
    )


def _parse_source_identity(
    *,
    smiles_stereo: str,
    smiles: str,
    inchi: str,
    source_inchikey: str,
    min_mw: float,
    max_mw: float,
    min_heavy_atoms: int,
    max_heavy_atoms: int,
    min_formal_charge: int,
    max_formal_charge: int,
) -> tuple[str, str, str, str, str]:
    if not source_inchikey:
        raise ValueError("missing_source_inchikey")
    candidates = [value for value in [smiles_stereo, smiles] if value]
    mismatches: list[str] = []
    selected_mol: Chem.Mol | None = None
    selected_smiles = ""
    route = ""
    for candidate in candidates:
        mol = Chem.MolFromSmiles(candidate)
        if mol is None:
            mismatches.append("smiles_parse_failed")
            continue
        key = Chem.MolToInchiKey(mol)
        if source_inchikey and key == source_inchikey:
            selected_mol = mol
            selected_smiles = candidate
            route = "source_stereo_smiles_inchikey_match" if candidate == smiles_stereo else "source_smiles_inchikey_match"
            break
        mismatches.append(f"smiles_inchikey_mismatch:{key or '<blank>'}")
    if selected_mol is None:
        if not inchi or not source_inchikey:
            raise ValueError(";".join(mismatches) or "missing_identity")
        mol = Chem.MolFromInchi(inchi)
        if mol is None:
            raise ValueError(";".join(mismatches + ["inchi_parse_failed"]))
        key = Chem.MolToInchiKey(mol)
        if key != source_inchikey:
            raise ValueError(";".join(mismatches + [f"inchi_inchikey_mismatch:{key or '<blank>'}"]))
        selected_mol = mol
        selected_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        route = "source_inchi_inchikey_match"
    if len(Chem.GetMolFrags(selected_mol)) != 1:
        raise ValueError("multi_component_ligand")
    if not any(atom.GetAtomicNum() == 6 for atom in selected_mol.GetAtoms()):
        raise ValueError("non_carbon_ligand")
    mw = float(Descriptors.MolWt(selected_mol))
    heavy = int(selected_mol.GetNumHeavyAtoms())
    charge = int(Chem.GetFormalCharge(selected_mol))
    if not (mw > min_mw and mw <= max_mw):
        raise ValueError(f"mw_out_of_bounds:{mw:.3f}")
    if not (min_heavy_atoms <= heavy <= max_heavy_atoms):
        raise ValueError(f"heavy_atom_count_out_of_bounds:{heavy}")
    if not (min_formal_charge <= charge <= max_formal_charge):
        raise ValueError(f"formal_charge_out_of_bounds:{charge}")
    parent = _standardize_mol(selected_smiles)
    canonical = Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True)
    standard_inchikey = Chem.MolToInchiKey(parent)
    if not standard_inchikey:
        raise ValueError("unable_to_derive_standard_inchikey")
    ligand_key, standardization_route = _structure_ligand_key(standard_inchikey, canonical)
    return selected_smiles, standard_inchikey, canonical, ligand_key, f"{route}|{standardization_route}"


def _struct_conn_excludes(instance: Mapping[str, Any], conns: list[Any]) -> str:
    ids = _map(instance.get("rcsb_nonpolymer_entity_instance_container_identifiers"))
    asym_ids = {_clean(ids.get("asym_id")), _clean(ids.get("auth_asym_id"))} - {""}
    for conn in conns:
        item = _map(conn)
        text = " ".join(_clean(item.get(key)).lower() for key in ("connect_type", "description", "role"))
        if not any(token in text for token in ("covale", "metal", "coord")):
            continue
        partners = [_map(item.get("connect_partner")), _map(item.get("connect_target"))]
        touched = False
        for partner in partners:
            if _clean(partner.get("label_asym_id")) in asym_ids or _clean(partner.get("auth_asym_id")) in asym_ids:
                touched = True
        if touched or not asym_ids:
            return _clean(item.get("connect_type")) or "struct_conn"
    return ""


def _add_exclusion(rows: list[dict[str, str]], entry: str, comp: str, inst: str, reason: str, detail: str = "") -> None:
    rows.append({"entry_id": entry, "component_id": comp, "instance_id": inst, "reason": reason, "detail": detail})


def _process_record(
    record: Mapping[str, Any],
    *,
    cutoff: str,
    clusters: Mapping[str, dict[str, str]],
    args: argparse.Namespace,
    exclusions: list[dict[str, str]],
) -> list[dict[str, Any]]:
    del cutoff
    entry_id = _clean(record.get("entry_id")).upper()
    release_date = _clean(record.get("initial_release_date"))[:10]
    entity_by_instance = _entity_by_instance(record)
    subject_entities, subject_instances = _subject_sets(record)
    target_lookup = _target_lookup(record)
    neighbors_by_instance = {
        _clean(item.get("instance_id")): _list(_map(item).get("neighbors"))
        for item in _list(record.get("rcsb_target_neighbors"))
    }
    doi, pmid, document_id, publication_key = _publication(record, entry_id)
    rows: list[dict[str, Any]] = []
    for instance_id, entity in sorted(entity_by_instance.items()):
        comp_id, smiles_stereo, smiles, inchi, source_inchikey, formula = _component_identity(entity)
        entity_id = _clean(entity.get("rcsb_id"))
        instance = next(
            (_map(candidate) for candidate in _list(entity.get("nonpolymer_entity_instances")) if _clean(_map(candidate).get("rcsb_id")) == instance_id),
            {},
        )
        if entity_id not in subject_entities and instance_id not in subject_instances:
            _add_exclusion(exclusions, entry_id, comp_id, instance_id, "not_pdb_native_subject_of_investigation")
            continue
        conn_detail = _struct_conn_excludes(instance, _list(record.get("rcsb_nonpolymer_struct_conn")))
        if conn_detail:
            _add_exclusion(exclusions, entry_id, comp_id, instance_id, "covalent_or_metal_coordination_struct_conn", conn_detail)
            continue
        target_to_residues: dict[str, set[str]] = defaultdict(set)
        max_distance = 0.0
        for neighbor in neighbors_by_instance.get(instance_id, []):
            item = _map(neighbor)
            try:
                distance = float(item.get("distance"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(distance) or distance > args.contact_distance:
                continue
            targets = _neighbor_targets(item, target_lookup)
            residue = "|".join(_clean(item.get(key)) for key in ("target_asym_id", "target_auth_seq_id", "target_seq_id", "target_comp_id"))
            for target in targets:
                target_to_residues[target].add(residue)
            max_distance = max(max_distance, distance)
        if len(target_to_residues) != 1:
            _add_exclusion(exclusions, entry_id, comp_id, instance_id, "not_exactly_one_contacted_human_target", json.dumps(sorted(target_to_residues)))
            continue
        uniprot, residues = next(iter(target_to_residues.items()))
        if uniprot not in clusters:
            _add_exclusion(exclusions, entry_id, comp_id, instance_id, "target_not_screenable", uniprot)
            continue
        residues = {value for value in residues if value.strip("|")}
        if len(residues) < args.min_contact_residues:
            _add_exclusion(exclusions, entry_id, comp_id, instance_id, "insufficient_unique_contact_residues", str(len(residues)))
            continue
        try:
            source_smiles, standard_inchikey, canonical, ligand_key, route = _parse_source_identity(
                smiles_stereo=smiles_stereo,
                smiles=smiles,
                inchi=inchi,
                source_inchikey=source_inchikey,
                min_mw=args.min_mw,
                max_mw=args.max_mw,
                min_heavy_atoms=args.min_heavy_atoms,
                max_heavy_atoms=args.max_heavy_atoms,
                min_formal_charge=args.min_formal_charge,
                max_formal_charge=args.max_formal_charge,
            )
        except ValueError as exc:
            _add_exclusion(exclusions, entry_id, comp_id, instance_id, "ligand_identity_or_chemistry_failed", str(exc))
            continue
        scaffold_smiles, scaffold_id = _scaffold(canonical)
        rows.append({
            "pair_id": _stable_hash(SCHEMA_VERSION, entry_id, comp_id, instance_id, ligand_key, uniprot),
            "entry_id": entry_id,
            "component_id": comp_id,
            "instance_id": instance_id,
            "initial_release_date": release_date,
            "uniprot": uniprot,
            "target_cluster_30": clusters[uniprot]["target_cluster_30"],
            "target_cluster_50": clusters[uniprot]["target_cluster_50"],
            "source_inchikey": source_inchikey,
            "source_identity_smiles": source_smiles,
            "standard_inchikey": standard_inchikey,
            "connectivity_key": standard_inchikey[:14],
            "canonical_smiles": canonical,
            "ligand_key": ligand_key,
            "standardization_route": "fragment_parent_canonical_smiles_key",
            "identity_route": route,
            "scaffold_smiles": scaffold_smiles,
            "scaffold_id": scaffold_id,
            "contact_distance_max_angstrom": max_distance,
            "contacted_residue_count": len(residues),
            "contacted_residues": sorted(residues),
            "structure_doi": doi,
            "structure_pmid": pmid,
            "source_document_id": document_id,
            "publication_key": publication_key,
            "evidence_date": release_date,
        })
    return rows


def _leakage_sets(frame: pd.DataFrame) -> dict[str, set[Any]]:
    targets = _clean_series(frame["uniprot"])
    ligand_keys = _benchmark_ligand_identity(frame)
    return {
        "pair": set(zip(targets, ligand_keys, strict=True)),
        "publication": set(_clean_series(frame["publication_key"])),
        "scaffold": set(_clean_series(frame["scaffold_id"])),
        "target_cluster_30": set(_clean_series(frame["target_cluster_30"])),
        "target_cluster_50": set(_clean_series(frame["target_cluster_50"])),
    }


def _prior_sets(train: pd.DataFrame, dev: pd.DataFrame) -> dict[str, dict[str, set[Any]]]:
    train_sets = _leakage_sets(train)
    dev_sets = _leakage_sets(dev)
    return {
        "train": train_sets,
        "prior": {
            key: train_sets[key] | dev_sets[key]
            for key in sorted(train_sets)
        },
    }


def _pair_absent(row: Any, prior_pairs: set[Any]) -> bool:
    target = _clean(row.uniprot)
    candidate_keys = {
        _clean(row.standard_inchikey).upper(),
        _clean(row.source_inchikey).upper(),
        _clean(row.ligand_key).upper(),
        _clean(row.canonical_smiles).upper(),
    }
    candidate_keys.discard("")
    return all((target, key) not in prior_pairs for key in candidate_keys)


def _apply_leakage(rows: pd.DataFrame, prior: Mapping[str, Mapping[str, set[Any]]]) -> pd.DataFrame:
    out = rows.copy()
    train = prior["train"]
    train_dev = prior["prior"]
    out["absent_pair_from_train"] = [
        _pair_absent(row, train["pair"]) for row in out.itertuples(index=False)
    ]
    out["absent_pair_from_prior_splits"] = [
        _pair_absent(row, train_dev["pair"])
        for row in out.itertuples(index=False)
    ]
    for column, suffix in [
        ("publication_key", "publication"),
        ("scaffold_id", "scaffold"),
        ("target_cluster_30", "target_cluster_30"),
        ("target_cluster_50", "target_cluster_50"),
    ]:
        values = out[column].map(_clean)
        out[f"absent_{suffix}_from_train"] = ~values.isin(train[suffix])
        out[f"absent_{suffix}_from_prior_splits"] = ~values.isin(train_dev[suffix])
    out["claimable"] = True
    out["is_dual_cold"] = out[sorted(DUAL_COLD_FLAGS)].all(axis=1)
    return out


def _dedupe_pairs(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows.reindex(columns=PAIR_COLUMNS)
    sorted_rows = rows.sort_values(["initial_release_date", "entry_id", "component_id", "instance_id", "ligand_key", "uniprot"], kind="mergesort")
    return sorted_rows.drop_duplicates(["ligand_key", "uniprot"], keep="first").reset_index(drop=True).reindex(columns=PAIR_COLUMNS)


def _ranking_queries(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame(columns=RANKING_COLUMNS)
    strict = pairs[pairs["is_dual_cold"].map(bool)].copy()
    rows: list[dict[str, Any]] = []
    for ligand_key, group in strict.groupby("ligand_key", sort=True):
        first = group.sort_values(["canonical_smiles", "standard_inchikey", "entry_id"], kind="mergesort").iloc[0]
        rows.append({
            "query_id": ligand_key,
            "ligand_key": ligand_key,
            "standard_inchikey": first["standard_inchikey"],
            "connectivity_key": first["connectivity_key"],
            "canonical_smiles": first["canonical_smiles"],
            "standardization_route": first["standardization_route"],
            "truth_targets": sorted(set(group["uniprot"].astype(str))),
            "n_truth_targets": int(group["uniprot"].nunique()),
            "split": "test",
            **{flag: True for flag in sorted(DUAL_COLD_FLAGS)},
            "source_entry_ids": sorted(set(group["entry_id"].astype(str))),
            "source_component_ids": sorted(set(group["component_id"].astype(str))),
            "source_instance_ids": sorted(set(group["instance_id"].astype(str))),
            "source_identity_routes": sorted(set(group["identity_route"].astype(str))),
            "source_publication_keys": sorted(set(group["publication_key"].astype(str))),
        })
    return pd.DataFrame(rows, columns=RANKING_COLUMNS)


def _adequacy(ranking: pd.DataFrame, *, args: argparse.Namespace) -> dict[str, Any]:
    target_counts: Counter[str] = Counter()
    for values in ranking["truth_targets"] if "truth_targets" in ranking else []:
        target_counts.update(str(value) for value in values)
    truth_pairs = sum(target_counts.values())
    fractions = {target: count / truth_pairs for target, count in target_counts.items()} if truth_pairs else {}
    max_fraction = max(fractions.values(), default=1.0)
    effective = 1.0 / sum(value * value for value in fractions.values()) if fractions else 0.0
    documents = sorted({key for values in ranking.get("source_publication_keys", []) for key in values})
    checks = {
        "min_queries": len(ranking) >= args.min_queries,
        "min_unique_truth_targets": len(target_counts) >= args.min_targets,
        "min_unique_source_documents": len(documents) >= args.min_documents,
        "max_truth_pair_target_fraction": max_fraction <= args.max_target_fraction,
        "min_effective_target_count": effective >= args.min_effective_targets,
    }
    return {
        "passes": all(checks.values()),
        "criteria": {
            "min_queries": args.min_queries,
            "min_unique_truth_targets": args.min_targets,
            "min_unique_source_documents": args.min_documents,
            "max_truth_pair_target_fraction": args.max_target_fraction,
            "min_effective_target_count": args.min_effective_targets,
        },
        "observed": {
            "queries": int(len(ranking)),
            "truth_pairs": int(truth_pairs),
            "unique_truth_targets": len(target_counts),
            "unique_source_documents": len(documents),
            "max_truth_pair_target_fraction": float(max_fraction),
            "effective_target_count": float(effective),
            "truth_pairs_by_target": dict(sorted(target_counts.items(), key=lambda item: (-item[1], item[0]))),
            "source_documents": documents,
        },
        "checks": checks,
    }


def _tmp_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def _remove_outputs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)
        _tmp_path(path).unlink(missing_ok=True)


def _write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    tmp.unlink(missing_ok=True)
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _write_csv_atomic(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    tmp.unlink(missing_ok=True)
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXCLUSION_COLUMNS)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: tuple(item[column] for column in EXCLUSION_COLUMNS)):
            writer.writerow(row)
    tmp.replace(path)


def _write_json_atomic(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    tmp.unlink(missing_ok=True)
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def build_panel(args: argparse.Namespace) -> dict[str, Any]:
    outputs = [args.out_pairs, args.out_exclusions, args.out_ranking_queries, args.out_manifest]
    _remove_outputs(outputs)
    source_manifest = _validate_source_manifest(args.source_manifest, args.raw_jsonl_gz)
    source_min_mw = float(source_manifest["filters"]["nonpolymer_molecular_weight_gt"])
    if not math.isclose(source_min_mw, args.min_mw, rel_tol=0.0, abs_tol=1e-12):
        raise PanelError(
            "--min-mw must exactly match the source snapshot molecular-weight search filter"
        )
    benchmark_manifest = _read_json(args.benchmark_manifest, "benchmark manifest")
    if benchmark_manifest.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        raise PanelError(f"benchmark manifest schema_version must be {BENCHMARK_SCHEMA_VERSION}")
    if args.train_parquet.resolve() == args.dev_parquet.resolve():
        raise PanelError("train and dev parquet paths must be distinct")
    train, train_meta = _validate_benchmark_parquet(args.train_parquet, benchmark_manifest, split="train", name="train.parquet")
    dev, dev_meta = _validate_benchmark_parquet(args.dev_parquet, benchmark_manifest, split="dev", name="dev.parquet")
    clusters, cluster_meta = _validate_clusters(args.screenable_target_clusters, args.screenable_target_cluster_manifest)
    exclusions: list[dict[str, str]] = []
    accepted: list[dict[str, Any]] = []
    cutoff = _clean(source_manifest.get("release_cutoff") or source_manifest.get("source_version", {}).get("entry_initial_release_date_gte"))
    for record in _stream_raw(args.raw_jsonl_gz, int(source_manifest["raw_jsonl_gz"]["rows"]), cutoff):
        accepted.extend(_process_record(record, cutoff=cutoff, clusters=clusters, args=args, exclusions=exclusions))
    pair_frame = _dedupe_pairs(_apply_leakage(pd.DataFrame(accepted), _prior_sets(train, dev)) if accepted else pd.DataFrame(columns=PAIR_COLUMNS))
    ranking = _ranking_queries(pair_frame)
    adequacy = _adequacy(ranking, args=args)
    _write_parquet_atomic(pair_frame, args.out_pairs)
    _write_csv_atomic(exclusions, args.out_exclusions)
    _write_parquet_atomic(ranking, args.out_ranking_queries)
    outputs_meta = {
        "pairs": {"path": str(args.out_pairs.resolve()), "sha256": _sha256(args.out_pairs), "rows": _parquet_rows(args.out_pairs)},
        "exclusions": {"path": str(args.out_exclusions.resolve()), "sha256": _sha256(args.out_exclusions), "rows": _csv_rows(args.out_exclusions)},
        "ranking_queries": {"path": str(args.out_ranking_queries.resolve()), "sha256": _sha256(args.out_ranking_queries), "rows": _parquet_rows(args.out_ranking_queries)},
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "inputs": {
            "raw_jsonl_gz": {"path": str(args.raw_jsonl_gz.resolve()), "sha256": _sha256(args.raw_jsonl_gz), "rows": int(source_manifest["raw_jsonl_gz"]["rows"])},
            "source_manifest": {"path": str(args.source_manifest.resolve()), "sha256": _sha256(args.source_manifest), "schema_version": SOURCE_SCHEMA_VERSION},
            "benchmark_manifest": {"path": str(args.benchmark_manifest.resolve()), "sha256": _sha256(args.benchmark_manifest), "schema_version": BENCHMARK_SCHEMA_VERSION},
            "train_parquet": train_meta,
            "dev_parquet": dev_meta,
            "screenable_target_clusters": cluster_meta,
        },
        "outputs": outputs_meta,
        "filters": {
            "pdb_native_subject_of_investigation": True,
            "initial_release_date_gte": cutoff,
            "human_target_mapping": "exactly one contacted human UniProt target",
            "contact_distance_lte_angstrom": args.contact_distance,
            "min_unique_contacted_target_residues": args.min_contact_residues,
            "exclude_covalent_or_metal_coordination_struct_conn": True,
            "mw_gt": args.min_mw,
            "mw_lte": args.max_mw,
            "carbon_containing": True,
            "heavy_atoms_min": args.min_heavy_atoms,
            "heavy_atoms_max": args.max_heavy_atoms,
            "formal_charge_min": args.min_formal_charge,
            "formal_charge_max": args.max_formal_charge,
        },
        "selection": {
            "accepted_evidence_pairs_before_dedupe": len(accepted),
            "accepted_evidence_pairs": int(len(pair_frame)),
            "strict_dual_cold_pairs": int(pair_frame["is_dual_cold"].sum()) if "is_dual_cold" in pair_frame else 0,
            "ranking_queries": {"test": {"rows": int(len(ranking)), "adequacy": adequacy}},
            "exclusions_by_reason": dict(sorted(Counter(row["reason"] for row in exclusions).items())),
        },
        "contract": {
            "positive_only": True,
            "no_inferred_negatives": True,
            "no_affinities_or_calibration": True,
            "never_training_or_model_selection": True,
            "ranking_split": "test",
            "all_ranking_rows_dual_cold": True,
            "pocket_leakage_audited": False,
            "pocket_cold_panel": False,
            "pocket_leakage_note": "Pocket leakage is NOT audited; this panel must not be described as pocket-cold.",
        },
        "passes_adequacy": bool(adequacy["passes"]),
    }
    _write_json_atomic(manifest, args.out_manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-jsonl-gz", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--train-parquet", required=True, type=Path)
    parser.add_argument("--dev-parquet", required=True, type=Path)
    parser.add_argument("--benchmark-manifest", required=True, type=Path)
    parser.add_argument("--screenable-target-clusters", required=True, type=Path)
    parser.add_argument("--screenable-target-cluster-manifest", required=True, type=Path)
    parser.add_argument("--out-pairs", required=True, type=Path)
    parser.add_argument("--out-exclusions", required=True, type=Path)
    parser.add_argument("--out-ranking-queries", required=True, type=Path)
    parser.add_argument("--out-manifest", required=True, type=Path)
    parser.add_argument("--min-mw", type=float, default=DEFAULT_MIN_MW)
    parser.add_argument("--max-mw", type=float, default=1000.0)
    parser.add_argument("--min-heavy-atoms", type=int, default=3)
    parser.add_argument("--max-heavy-atoms", type=int, default=100)
    parser.add_argument("--min-formal-charge", type=int, default=-3)
    parser.add_argument("--max-formal-charge", type=int, default=3)
    parser.add_argument("--contact-distance", type=float, default=4.5)
    parser.add_argument("--min-contact-residues", type=int, default=3)
    parser.add_argument("--min-queries", type=int, default=100)
    parser.add_argument("--min-targets", type=int, default=20)
    parser.add_argument("--min-documents", type=int, default=20)
    parser.add_argument("--max-target-fraction", type=float, default=0.20)
    parser.add_argument("--min-effective-targets", type=float, default=10.0)
    args = parser.parse_args(argv)
    if args.min_mw < 0 or args.max_mw <= args.min_mw:
        raise SystemExit("MW bounds must satisfy 0 <= min < max")
    if args.min_heavy_atoms < 1 or args.max_heavy_atoms < args.min_heavy_atoms:
        raise SystemExit("heavy atom bounds are invalid")
    if args.contact_distance <= 0 or args.min_contact_residues < 1:
        raise SystemExit("contact distance/residue bounds are invalid")
    if args.min_queries < 0 or args.min_targets < 0 or args.min_documents < 0:
        raise SystemExit("adequacy thresholds must be non-negative")
    if not (0 < args.max_target_fraction <= 1):
        raise SystemExit("max target fraction must be in (0, 1]")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = build_panel(args)
    except PanelError as exc:
        _remove_outputs([args.out_pairs, args.out_exclusions, args.out_ranking_queries, args.out_manifest])
        raise SystemExit(str(exc)) from exc
    print(
        "Built RCSB holo panel: "
        f"pairs={manifest['outputs']['pairs']['rows']} "
        f"queries={manifest['outputs']['ranking_queries']['rows']} "
        f"passes_adequacy={manifest['passes_adequacy']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
