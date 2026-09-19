#!/usr/bin/env python3
"""Rank pharmacophore and target-supported substitute hypotheses for one compound.

The workflow is deliberately claim-limited. It combines a ligand-feature
pharmacophore similarity with same-target public activity evidence, structural
alerts, cosmetic-reference status, property suitability, and a routeability
proxy. It does not treat any candidate as experimentally validated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import shutil
import statistics
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs, RDConfig
from rdkit.Chem import (
    QED,
    AllChem,
    ChemicalFeatures,
    Crippen,
    Descriptors,
    FilterCatalog,
    Lipinski,
    rdFingerprintGenerator,
    rdFMCS,
    rdMolDescriptors,
)
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D

try:
    from .interaction_anchor import (
        PRESERVATION_BASIS,
        InteractionAnchorError,
        load_anchor_map,
        score_anchor_preservation,
        validate_anchor_map,
    )
except ImportError:  # Executed as a standalone script.
    from interaction_anchor import (  # type: ignore[no-redef]
        PRESERVATION_BASIS,
        InteractionAnchorError,
        load_anchor_map,
        score_anchor_preservation,
        validate_anchor_map,
    )

try:
    from .compound_applicability import assess, physchem, refusal_message
except ImportError:  # Executed as a standalone script.
    from compound_applicability import (  # type: ignore[no-redef]
        assess,
        physchem,
        refusal_message,
    )


SCHEMA_VERSION = "skinscout.substitute_discovery.v2"
DEFAULT_MATERIAL_TRACK_FRACTION = 0.40
DEFAULT_PHARMACOPHORE_TRACK_FRACTION = 0.20
DEFAULT_MAX_3D_CONFORMERS = 50
DEFAULT_MAX_3D_CANDIDATES = 128
DEFAULT_MIN_3D_FEATURE_RECALL = 0.80
DEFAULT_MAX_FEATURE_DISTANCE_RMSD = 1.50
DEFAULT_MIN_ANCHOR_PRESERVATION = 0.80
UNIPROT_ACCESSION_RE = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|"
    r"[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?:-[0-9]+)?$"
)
DEFAULT_ACTIVITY_EVIDENCE = (
    Path("data/chembl37/activity_evidence.parquet"),
    Path("data/bindingdb/evidence_v1/activity_evidence.parquet"),
    Path("data/gtopdb/evidence_v1/activity_evidence.parquet"),
)
DEFAULT_ALIAS_EVIDENCE = (
    Path("data/discovery_aliases/sources/chembl_aliases.parquet"),
    Path("data/discovery_aliases/sources/gtopdb_aliases.parquet"),
)
OUTPUT_FILES = (
    "substitute_report.json",
    "substitute_candidates.csv",
    "substitute_candidates_3d.sdf",
    "substitute_report.html",
    "substitute_report.md",
)
TARGET_COLUMNS = ("uniprot", "target_uniprot", "accession", "target_accession")
SMILES_COLUMNS = ("smiles", "ligand_smiles", "molecule_smiles", "canonical_smiles")
INCHIKEY_COLUMNS = (
    "standard_inchi_key",
    "ligand_inchikey",
    "molecule_inchikey",
    "inchikey",
)
LIGAND_ID_COLUMNS = (
    "molecule_chembl_id",
    "ligand_id",
    "gtopdb_ligand_id",
    "molecule_id",
)
PACTIVITY_COLUMNS = ("pchembl", "pchembl_value", "affinity_pvalue")
VALUE_COLUMNS = ("affinity_value", "activity_value", "standard_value", "act_value")
UNIT_COLUMNS = ("affinity_unit", "activity_unit", "standard_units", "act_units")
RELATION_COLUMNS = ("relation", "standard_relation")
CENSOR_COLUMNS = ("censor",)
SOURCE_COLUMNS = ("source_db", "source_name", "database", "source")
RELEASE_COLUMNS = ("source_release", "source_version", "release", "version")
DOI_COLUMNS = ("source_doi", "doi")
PMID_COLUMNS = ("source_pmid", "pubmed_id", "pmid")
ACTIVITY_TYPE_COLUMNS = ("affinity_type", "activity_type", "standard_type", "act_type")
FEATURE_FAMILIES = (
    "Donor",
    "Acceptor",
    "Aromatic",
    "Hydrophobe",
    "PosIonizable",
    "NegIonizable",
)
ALERT_CATALOGS = {
    "PAINS_A": FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A,
    "PAINS_B": FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B,
    "PAINS_C": FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C,
    "BRENK": FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK,
    "NIH": FilterCatalog.FilterCatalogParams.FilterCatalogs.NIH,
}
UNIT_TO_MOLAR = {
    "m": 1.0,
    "mm": 1e-3,
    "um": 1e-6,
    "micromolar": 1e-6,
    "nm": 1e-9,
    "nanomolar": 1e-9,
    "pm": 1e-12,
    "picomolar": 1e-12,
}


@dataclass(frozen=True)
class MoleculeIdentity:
    mol: Chem.Mol
    smiles: str
    inchikey: str


@dataclass
class CandidateSeed:
    smiles: str
    inchikey: str
    names: set[str] = field(default_factory=set)
    functions: set[str] = field(default_factory=set)
    sources: set[str] = field(default_factory=set)
    cosing_reference: bool = False
    activities: list[dict[str, Any]] = field(default_factory=list)
    preferred_name: str | None = None


@dataclass(frozen=True)
class TargetAnchorContext:
    parent_mol: Chem.Mol
    parent_atom_indices: list[int]
    target_id: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _clean(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.upper() in {"", "NA", "N/A", "NAN", "NONE", "NULL"} else text


def _finite(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"{label} is required and must be non-empty: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _require_new_outputs(out_dir: Path) -> None:
    for filename in OUTPUT_FILES:
        path = out_dir / filename
        if path.exists() or path.is_symlink():
            raise SystemExit(
                f"refusing to overwrite pre-existing substitute artifact: {path}"
            )


def _standardize_smiles(smiles: str, label: str) -> MoleculeIdentity:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"{label} is not valid SMILES")
    try:
        molecule = rdMolStandardize.Cleanup(molecule)
        molecule = rdMolStandardize.FragmentParent(molecule)
        molecule = rdMolStandardize.Uncharger().uncharge(molecule)
        Chem.SanitizeMol(molecule)
    except Exception as exc:
        raise ValueError(f"{label} could not be standardized: {exc}") from exc
    if molecule.GetNumHeavyAtoms() == 0:
        raise ValueError(f"{label} has no heavy atoms after standardization")
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    return MoleculeIdentity(
        mol=molecule,
        smiles=canonical,
        inchikey=Chem.MolToInchiKey(molecule),
    )


def _target_anchor_context(
    parent: MoleculeIdentity,
    payload: dict[str, Any] | None,
    target_id: str | None,
) -> TargetAnchorContext | None:
    if payload is None:
        return None
    if not target_id:
        raise InteractionAnchorError(
            "A target ID is required when interaction anchors are supplied"
        )
    validate_anchor_map(
        payload,
        path=Path("<in-memory-interaction-anchor-map>"),
        expected_parent_inchikey=parent.inchikey,
        required_target=target_id,
    )
    parent_record = payload["parent"]
    canonical_smiles = parent_record.get("canonical_smiles")
    target_record = payload["targets"][target_id]
    anchor_indices = target_record.get("confirmed_canonical_parent_atoms")
    if not isinstance(canonical_smiles, str) or not canonical_smiles:
        raise InteractionAnchorError(
            "Interaction-anchor map lacks canonical parent SMILES"
        )
    if not isinstance(anchor_indices, list) or not anchor_indices:
        raise InteractionAnchorError(
            f"Interaction-anchor map lacks canonical parent atoms for target {target_id}"
        )
    anchor_parent = Chem.MolFromSmiles(canonical_smiles)
    if anchor_parent is None or Chem.MolToInchiKey(anchor_parent) != parent.inchikey:
        raise InteractionAnchorError(
            "Interaction-anchor canonical parent does not match the requested compound"
        )
    return TargetAnchorContext(
        parent_mol=anchor_parent,
        parent_atom_indices=list(anchor_indices),
        target_id=target_id,
    )


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"Candidate library is required and must be non-empty: {path}")
    try:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        if path.suffix.lower() in {".tsv", ".txt"}:
            return pd.read_csv(path, sep="\t")
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
    except Exception as exc:
        raise SystemExit(f"Candidate library failed to parse: {path}: {exc}") from exc
    raise SystemExit(f"Unsupported candidate library format: {path}")


def _candidate_library(
    path: Path,
) -> tuple[dict[str, CandidateSeed], dict[str, int], dict[str, Any]]:
    table = _read_table(path)
    by_lower = {str(column).lower(): str(column) for column in table.columns}
    smiles_col = next((by_lower[name] for name in SMILES_COLUMNS if name in by_lower), None)
    if smiles_col is None:
        raise SystemExit(f"Candidate library is missing a SMILES column: {path}")
    name_col = next(
        (by_lower[name] for name in ("inci_name", "name", "compound_name", "label") if name in by_lower),
        None,
    )
    functions_col = next(
        (by_lower[name] for name in ("functions", "function", "cosmetic_functions") if name in by_lower),
        None,
    )
    seeds: dict[str, CandidateSeed] = {}
    counts = Counter(total_rows=len(table), accepted_rows=0, invalid_rows=0, duplicate_rows=0)
    for row_index, row in table.iterrows():
        raw_smiles = _clean(row.get(smiles_col))
        if not raw_smiles:
            counts["invalid_rows"] += 1
            continue
        try:
            identity = _standardize_smiles(raw_smiles, f"candidate row {row_index}")
        except ValueError:
            counts["invalid_rows"] += 1
            continue
        seed = seeds.get(identity.smiles)
        if seed is None:
            seed = CandidateSeed(
                smiles=identity.smiles,
                inchikey=identity.inchikey,
                cosing_reference=True,
            )
            seeds[identity.smiles] = seed
            counts["accepted_rows"] += 1
        else:
            counts["duplicate_rows"] += 1
        seed.sources.add("cosing")
        if name_col and (name := _clean(row.get(name_col))):
            seed.names.add(name)
        if functions_col and (functions := _clean(row.get(functions_col))):
            seed.functions.update(item.strip() for item in functions.split(";") if item.strip())
    if not seeds:
        raise SystemExit(f"Candidate library contains no standardizable molecules: {path}")
    fingerprint = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    return seeds, dict(counts), fingerprint


def _identifier_like_name(value: str) -> bool:
    compact = value.strip().upper()
    return bool(
        re.fullmatch(
            r"(?:CHEMBL\d+|GTOPDB:?\d+|PUBCHEM\s+CID\s+\d+|\d+)",
            compact,
        )
    )


def _alias_name_index(
    paths: list[Path],
    accepted_inchikeys: set[str],
) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    if not paths or not accepted_inchikeys:
        return {}, []
    accepted = {value.upper() for value in accepted_inchikeys}
    counts: dict[str, Counter[str]] = {}
    forms: dict[str, dict[str, str]] = {}
    audits: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists() or not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"Alias evidence is required and must be non-empty: {path}")
        try:
            parquet = pq.ParquetFile(path)
        except Exception as exc:
            raise SystemExit(f"Alias evidence failed to parse: {path}: {exc}") from exc
        columns = set(parquet.schema_arrow.names)
        if not {"alias", "inchikey"}.issubset(columns):
            raise SystemExit(
                f"Alias evidence is missing required fields alias/inchikey: {path}"
            )
        try:
            table = pq.read_table(
                path,
                columns=["alias", "inchikey"],
                filters=[("inchikey", "in", sorted(accepted))],
            )
        except Exception as exc:
            raise SystemExit(f"Alias evidence query failed: {path}: {exc}") from exc
        matched_rows = 0
        for row in table.to_pylist():
            inchikey = _clean(row.get("inchikey")).upper()
            alias = _clean(row.get("alias"))
            if inchikey not in accepted or not alias or len(alias) > 240:
                continue
            normalized = alias.casefold()
            counts.setdefault(inchikey, Counter())[normalized] += 1
            forms.setdefault(inchikey, {}).setdefault(normalized, alias)
            matched_rows += 1
        audits.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "matched_rows": matched_rows,
                "matched_inchikeys": len(
                    {
                        _clean(value).upper()
                        for value in table.column("inchikey").to_pylist()
                        if _clean(value).upper() in accepted
                    }
                ),
                "identity_contract": "exact_full_inchikey",
            }
        )

    ranked: dict[str, list[str]] = {}
    for inchikey, aliases in counts.items():
        ordered = sorted(
            aliases,
            key=lambda normalized: (
                -aliases[normalized],
                _identifier_like_name(forms[inchikey][normalized]),
                "<" in forms[inchikey][normalized]
                or "&" in forms[inchikey][normalized],
                len(forms[inchikey][normalized]),
                normalized,
            ),
        )
        ranked[inchikey] = [forms[inchikey][value] for value in ordered[:20]]
    return ranked, audits


def _enrich_seed_names(
    seeds: dict[str, CandidateSeed],
    paths: list[Path],
) -> list[dict[str, Any]]:
    aliases, audits = _alias_name_index(
        paths,
        {seed.inchikey for seed in seeds.values()},
    )
    for seed in seeds.values():
        names = aliases.get(seed.inchikey, [])
        if not names:
            continue
        seed.preferred_name = names[0]
        seed.names.update(names)
    return audits


def _column(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    by_lower = {str(value).strip().lower(): str(value) for value in columns}
    return next((by_lower[alias.lower()] for alias in aliases if alias.lower() in by_lower), None)


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value) != 0.0
    return _clean(value).lower() in {"1", "true", "yes", "y", "censored"}


def _normalized_unit(value: object) -> str:
    return (
        _clean(value)
        .lower()
        .replace("μ", "u")
        .replace("µ", "u")
        .replace("mol/l", "m")
        .replace("molar", "m")
        .replace(" ", "")
    )


def _activity_value(row: dict[str, object], mapped: dict[str, str | None]) -> float | None:
    relation_col = mapped["relation"]
    relation = _clean(row.get(relation_col)) if relation_col else ""
    if relation not in {"=", "=="}:
        return None
    censor_col = mapped["censor"]
    if censor_col and _truthy(row.get(censor_col)):
        return None
    pactivity_col = mapped["pactivity"]
    if pactivity_col:
        value = _finite(row.get(pactivity_col))
        if value is not None and 0.0 <= value <= 15.0:
            return value
    value_col = mapped["value"]
    unit_col = mapped["unit"]
    if not value_col or not unit_col:
        return None
    value = _finite(row.get(value_col))
    factor = UNIT_TO_MOLAR.get(_normalized_unit(row.get(unit_col)))
    if value is None or value <= 0.0 or factor is None:
        return None
    pactivity = -math.log10(value * factor)
    return pactivity if 0.0 <= pactivity <= 15.0 else None


def _mapped_activity_columns(path: Path, columns: list[str]) -> dict[str, str | None]:
    mapped = {
        "target": _column(columns, TARGET_COLUMNS),
        "smiles": _column(columns, SMILES_COLUMNS),
        "inchikey": _column(columns, INCHIKEY_COLUMNS),
        "ligand_id": _column(columns, LIGAND_ID_COLUMNS),
        "pactivity": _column(columns, PACTIVITY_COLUMNS),
        "value": _column(columns, VALUE_COLUMNS),
        "unit": _column(columns, UNIT_COLUMNS),
        "relation": _column(columns, RELATION_COLUMNS),
        "censor": _column(columns, CENSOR_COLUMNS),
        "source": _column(columns, SOURCE_COLUMNS),
        "release": _column(columns, RELEASE_COLUMNS),
        "doi": _column(columns, DOI_COLUMNS),
        "pmid": _column(columns, PMID_COLUMNS),
        "activity_type": _column(columns, ACTIVITY_TYPE_COLUMNS),
    }
    missing = [key for key in ("target", "smiles", "relation") if not mapped[key]]
    if not mapped["pactivity"] and not (mapped["value"] and mapped["unit"]):
        missing.append("potency")
    if missing:
        raise SystemExit(
            f"Activity evidence is missing required normalized fields {missing}: {path}"
        )
    return mapped


def _activity_observation(
    row: dict[str, object],
    mapped: dict[str, str | None],
    pactivity: float,
    fallback_source: str,
) -> dict[str, Any]:
    def value(name: str) -> str:
        column = mapped[name]
        return _clean(row.get(column)) if column else ""

    source = value("source") or fallback_source
    return {
        "pactivity": round(pactivity, 6),
        "source": source,
        "release": value("release"),
        "activity_type": value("activity_type").upper().replace(" ", ""),
        "doi": value("doi"),
        "pmid": value("pmid"),
    }


def _target_activity_candidates(
    paths: list[Path],
    target_id: str,
    parent: MoleculeIdentity,
    *,
    max_direct_molecules: int,
) -> tuple[dict[str, CandidateSeed], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, dict[str, Any]] = {}
    source_audits: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists() or not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"Activity evidence is required and must be non-empty: {path}")
        try:
            parquet = pq.ParquetFile(path)
        except Exception as exc:
            raise SystemExit(f"Activity evidence failed to parse: {path}: {exc}") from exc
        columns = list(parquet.schema_arrow.names)
        mapped = _mapped_activity_columns(path, columns)
        selected_columns = list(dict.fromkeys(value for value in mapped.values() if value))
        target_rows = 0
        usable_rows = 0
        invalid_smiles_rows = 0
        identity_mismatch_rows = 0
        identity_detail_mismatch_rows = 0
        fallback_source = path.parent.name or path.stem
        for batch in parquet.iter_batches(columns=selected_columns, batch_size=131_072):
            target_array = batch.column(batch.schema.get_field_index(str(mapped["target"])))
            filtered = batch.filter(pc.equal(target_array, target_id))
            if filtered.num_rows == 0:
                continue
            target_rows += filtered.num_rows
            payload = filtered.to_pydict()
            for row_index in range(filtered.num_rows):
                row = {name: payload[name][row_index] for name in selected_columns}
                pactivity = _activity_value(row, mapped)
                raw_smiles = _clean(row.get(str(mapped["smiles"])))
                if pactivity is None or not raw_smiles:
                    continue
                raw_molecule = Chem.MolFromSmiles(raw_smiles)
                if raw_molecule is None:
                    invalid_smiles_rows += 1
                    continue
                raw_computed_key = Chem.MolToInchiKey(raw_molecule)
                try:
                    identity = _standardize_smiles(raw_smiles, "activity ligand")
                except ValueError:
                    invalid_smiles_rows += 1
                    continue
                raw_key = (
                    _clean(row.get(str(mapped["inchikey"])))
                    if mapped["inchikey"]
                    else ""
                )
                if raw_key:
                    supplied_key = raw_key.upper()
                    supplied_connectivity = supplied_key.split("-", 1)[0]
                    accepted_connectivity = {
                        raw_computed_key.split("-", 1)[0],
                        identity.inchikey.split("-", 1)[0],
                    }
                    if supplied_connectivity not in accepted_connectivity:
                        identity_mismatch_rows += 1
                        continue
                    if supplied_key not in {raw_computed_key, identity.inchikey}:
                        identity_detail_mismatch_rows += 1
                        continue
                grouping_key = identity.smiles
                bucket = grouped.setdefault(
                    grouping_key,
                    {
                        "smiles": identity.smiles,
                        "inchikey": identity.inchikey,
                        "activities": [],
                        "labels": set(),
                    },
                )
                ligand_id_col = mapped["ligand_id"]
                if ligand_id_col and (ligand_id := _clean(row.get(ligand_id_col))):
                    bucket["labels"].add(ligand_id)
                bucket["activities"].append(
                    _activity_observation(row, mapped, pactivity, fallback_source)
                )
                usable_rows += 1
        source_audits.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "target_rows": target_rows,
                "usable_exact_potency_rows": usable_rows,
                "invalid_smiles_rows": invalid_smiles_rows,
                "identity_mismatch_rows": identity_mismatch_rows,
                "identity_detail_mismatch_rows": identity_detail_mismatch_rows,
                "mapped_columns": mapped,
            }
        )

    def median_pactivity(item: tuple[str, dict[str, Any]]) -> tuple[float, str]:
        key, bucket = item
        values = [float(record["pactivity"]) for record in bucket["activities"]]
        return statistics.median(values), key

    ordered = sorted(grouped.items(), key=median_pactivity, reverse=True)
    selected = ordered[:max_direct_molecules]
    selected_keys = {key for key, _ in selected}
    for key, bucket in ordered[max_direct_molecules:]:
        if bucket.get("inchikey") == parent.inchikey and key not in selected_keys:
            selected.append((key, bucket))
            selected_keys.add(key)

    seeds: dict[str, CandidateSeed] = {}
    for _key, bucket in selected:
        seed = seeds.setdefault(
            str(bucket["smiles"]),
            CandidateSeed(
                smiles=str(bucket["smiles"]),
                inchikey=str(bucket["inchikey"]),
            ),
        )
        seed.sources.update(record["source"] for record in bucket["activities"])
        seed.names.update(str(label) for label in bucket["labels"])
        seed.activities.extend(bucket["activities"])
    summary = [
        {
            "target_id": target_id,
            "unique_direct_molecules": len(grouped),
            "selected_direct_molecules": len(seeds),
            "rejected_unstandardizable": sum(
                int(audit["invalid_smiles_rows"]) for audit in source_audits
            ),
            "rejected_identity_mismatch": sum(
                int(audit["identity_mismatch_rows"])
                + int(audit["identity_detail_mismatch_rows"])
                for audit in source_audits
            ),
            "rejected_identity_detail_mismatch": sum(
                int(audit["identity_detail_mismatch_rows"])
                for audit in source_audits
            ),
            "selection_cap": max_direct_molecules,
        }
    ]
    return seeds, source_audits, summary


def _merge_seed(destination: CandidateSeed, source: CandidateSeed) -> None:
    destination.names.update(source.names)
    destination.functions.update(source.functions)
    destination.sources.update(source.sources)
    destination.cosing_reference = destination.cosing_reference or source.cosing_reference
    destination.activities.extend(source.activities)
    if destination.preferred_name is None:
        destination.preferred_name = source.preferred_name


def _feature_counts(molecule: Chem.Mol, factory: Any) -> Counter[str]:
    counts: Counter[str] = Counter()
    for feature in factory.GetFeaturesForMol(molecule):
        family = feature.GetFamily()
        if family in FEATURE_FAMILIES:
            counts[family] += 1
    return counts


def _feature_overlap_scores(
    parent: Counter[str], candidate: Counter[str]
) -> tuple[float, float, float]:
    parent_total = sum(parent.values())
    candidate_total = sum(candidate.values())
    if parent_total == 0 or candidate_total == 0:
        return 0.0, 0.0, 0.0
    overlap = sum(
        min(count, candidate.get(family, 0)) for family, count in parent.items()
    )
    recall = overlap / parent_total
    precision = overlap / candidate_total
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return recall, precision, f1


def _distance(point_a: Any, point_b: Any) -> float:
    return math.sqrt(
        (float(point_a.x) - float(point_b.x)) ** 2
        + (float(point_a.y) - float(point_b.y)) ** 2
        + (float(point_a.z) - float(point_b.z)) ** 2
    )


def _conformer_ensemble(
    molecule: Chem.Mol,
    *,
    seed: int,
    max_conformers: int,
) -> tuple[Chem.Mol | None, list[int], str]:
    if max_conformers <= 0:
        return None, [], "disabled"
    working = Chem.AddHs(Chem.Mol(molecule))
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.useRandomCoords = False
    params.pruneRmsThresh = 0.10
    params.numThreads = 1
    try:
        conf_ids = list(
            AllChem.EmbedMultipleConfs(
                working,
                numConfs=int(max_conformers),
                params=params,
            )
        )
    except Exception:
        return None, [], "etkdg_v3_failed"
    if not conf_ids:
        return None, [], "etkdg_v3_failed"

    optimization_status = "etkdg_v3_mmff"
    properties = AllChem.MMFFGetMoleculeProperties(working, mmffVariant="MMFF94s")
    if properties is not None:
        try:
            results = AllChem.MMFFOptimizeMoleculeConfs(
                working,
                numThreads=1,
                mmffVariant="MMFF94s",
                maxIters=250,
            )
        except Exception:
            results = []
            optimization_status = "etkdg_v3_mmff_failed"
    else:
        results = []
        optimization_status = "etkdg_v3_mmff_unavailable"

    # MMFF가 돌지 않았거나(주석 화합물처럼 파라미터가 없는 원소) 결과 수가 맞지
    # 않으면 에너지 없이 임베딩 순서를 그대로 쓴다. 위 두 갈래가 `results = []`로
    # 두고 상태 문자열까지 준비해 두었는데, 바로 아래 strict=True zip이 그 자리에서
    # 터져서 그 상태가 반환된 적이 없었다. 등재 원료 507종 중 3종(유기주석)이 실제로
    # 여기서 죽는다.
    if results and len(results) != len(conf_ids):
        optimization_status = "etkdg_v3_mmff_partial"
        results = []

    energies: dict[int, float] = {}
    for conf_id, result in zip(conf_ids, results, strict=True) if results else ():
        if (
            isinstance(result, tuple)
            and len(result) >= 2
            and math.isfinite(float(result[1]))
        ):
            energies[int(conf_id)] = float(result[1])
    ordered = sorted(conf_ids, key=lambda conf_id: (energies.get(int(conf_id), math.inf), int(conf_id)))
    return working, [int(conf_id) for conf_id in ordered], optimization_status


def _feature_records(
    molecule: Chem.Mol,
    factory: Any,
    conf_id: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for feature in factory.GetFeaturesForMol(molecule):
        family = feature.GetFamily()
        if family not in FEATURE_FAMILIES:
            continue
        atom_ids = tuple(int(atom_id) for atom_id in feature.GetAtomIds())
        records.append(
            {
                "family": family,
                "atom_ids": atom_ids,
                "position": feature.GetPos(conf_id),
            }
        )
    return sorted(records, key=lambda item: (str(item["family"]), tuple(item["atom_ids"])))


def _parent_feature_records_by_conformer(
    parent_ensemble: tuple[Chem.Mol | None, list[int], str],
    factory: Any,
) -> dict[int, list[dict[str, Any]]]:
    parent_3d, parent_conf_ids, _status = parent_ensemble
    if parent_3d is None:
        return {}
    return {
        int(conf_id): _feature_records(parent_3d, factory, int(conf_id))
        for conf_id in parent_conf_ids
    }


MCS_ATOM_MAP_TIMEOUT_SECONDS = 5.0


def _mcs_atom_map(
    parent: Chem.Mol,
    candidate: Chem.Mol,
    *,
    timeout_seconds: float | None = None,
) -> tuple[list[tuple[int, int]], bool]:
    """MCS 원자맵과 시간 초과 여부를 함께 돌려준다.

    RDKit의 MCS timeout 은 1초 단위라, 남은 예산이 0보다 크면 다음 초로 올림해
    넘긴다. `canceled` 는 시간 초과이지 "공통 구조 없음"이 아니므로 호출자가
    구분할 수 있게 두 번째 값으로 알린다.
    """
    budget = (
        MCS_ATOM_MAP_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    )
    if budget <= 0:
        return [], True
    result = rdFMCS.FindMCS(
        [parent, candidate],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrderExact,
        matchChiralTag=True,
        timeout=max(1, int(math.ceil(min(budget, MCS_ATOM_MAP_TIMEOUT_SECONDS)))),
    )
    if result.canceled:
        return [], True
    if not result.smartsString:
        return [], False
    query = Chem.MolFromSmarts(result.smartsString)
    if query is None:
        return [], False
    parent_match = parent.GetSubstructMatch(query)
    candidate_match = candidate.GetSubstructMatch(query)
    if not parent_match or not candidate_match:
        return [], False
    return [(int(candidate_atom), int(parent_atom)) for parent_atom, candidate_atom in zip(parent_match, candidate_match, strict=True)], False


def _matched_feature_rmsd(
    parent_features: list[dict[str, Any]],
    candidate_features: list[dict[str, Any]],
) -> tuple[int, float | None]:
    unmatched = set(range(len(candidate_features)))
    squared_distances: list[float] = []
    for parent_feature in parent_features:
        same_family = [
            index
            for index in unmatched
            if candidate_features[index]["family"] == parent_feature["family"]
        ]
        if not same_family:
            continue
        distance, _atom_ids, best_index = min(
            (
                _distance(
                    parent_feature["position"],
                    candidate_features[index]["position"],
                ),
                tuple(candidate_features[index]["atom_ids"]),
                index,
            )
            for index in same_family
        )
        unmatched.remove(best_index)
        squared_distances.append(distance * distance)
    if len(squared_distances) < 3:
        return len(squared_distances), None
    return len(squared_distances), math.sqrt(sum(squared_distances) / len(squared_distances))


def _pharmacophore_3d_comparison(
    parent_mol: Chem.Mol,
    candidate_mol: Chem.Mol,
    factory: Any,
    parent_ensemble: tuple[Chem.Mol | None, list[int], str],
    *,
    seed: int,
    max_conformers: int,
    parent_features_by_conformer: dict[int, list[dict[str, Any]]] | None = None,
    deadline: float | None = None,
) -> dict[str, Any]:
    """`deadline` 은 `time.monotonic()` 기준 마감 시각이다.

    호출자의 예산이 "이 쌍을 시작해도 되는가"만 물으면 상한이 지켜지지 않는다.
    한 쌍이 후보 임베딩(실측 최대 20초 남짓)과 컨포머 쌍 100회 정렬을 다 돌기
    때문이다. 마감을 안쪽까지 내려보내 도중에도 멈출 수 있게 한다.
    """
    import time as _time

    def _out_of_time() -> bool:
        return deadline is not None and _time.monotonic() >= deadline

    parent_3d, parent_conf_ids, parent_status = parent_ensemble
    # 부모 앙상블이 없으면 결과는 후보를 보기 전에 이미 정해져 있다. 후보를
    # 먼저 임베딩하면 결론이 난 계산에 후보마다 ETKDG+MMFF 비용을 낸다 -
    # 실측으로 25행에 31초를 쓰고 전부 빈 칸이 나왔다.
    if parent_3d is None or not parent_conf_ids:
        return {
            "status": "parent_unavailable",
            "basis": "etkdg_v3_mmff_ligand_feature_alignment",
            "parent_conformer_count": len(parent_conf_ids),
            "candidate_conformer_count": 0,
            "parent_conformer_status": parent_status,
            "candidate_conformer_status": "not_attempted",
            "matched_feature_count": 0,
            "parent_feature_count": 0,
            "feature_family_recall": None,
            "feature_distance_rmsd": None,
        }
    if _out_of_time():
        return {
            "status": "budget_exhausted",
            "basis": "etkdg_v3_mmff_ligand_feature_alignment",
            "parent_conformer_count": len(parent_conf_ids),
            "candidate_conformer_count": 0,
            "parent_conformer_status": parent_status,
            "candidate_conformer_status": "not_attempted",
            "matched_feature_count": 0,
            "parent_feature_count": 0,
            "feature_family_recall": None,
            "feature_distance_rmsd": None,
        }
    candidate_3d, candidate_conf_ids, candidate_status = _conformer_ensemble(
        candidate_mol,
        seed=seed,
        max_conformers=max_conformers,
    )
    if candidate_3d is None or not candidate_conf_ids:
        return {
            "status": "unavailable",
            "basis": "etkdg_v3_mmff_ligand_feature_alignment",
            "parent_conformer_count": len(parent_conf_ids),
            "candidate_conformer_count": len(candidate_conf_ids),
            "parent_conformer_status": parent_status,
            "candidate_conformer_status": candidate_status,
            "matched_feature_count": 0,
            "parent_feature_count": 0,
            "feature_family_recall": None,
            "feature_distance_rmsd": None,
        }
    # 후보 임베딩(ETKDG+MMFF)은 RDKit 호출 하나라 도중에 멈출 수 없다. 생성이
    # 끝난 뒤 남은 예산을 다시 본다. 예전에는 이 확인 없이 MCS와 정렬을 이어서
    # 돌렸고, 마감을 넘긴 결과가 "공통 구조 없음"과 같은 unavailable 로 나왔다.
    remaining_budget: float | None = None
    if deadline is not None:
        remaining_budget = deadline - _time.monotonic()
        if remaining_budget <= 0:
            return {
                "status": "budget_exhausted",
                "basis": "etkdg_v3_mmff_ligand_feature_alignment",
                "parent_conformer_count": len(parent_conf_ids),
                "candidate_conformer_count": len(candidate_conf_ids),
                "evaluated_pairs": 0,
                "total_pairs": len(parent_conf_ids) * len(candidate_conf_ids),
                "parent_conformer_status": parent_status,
                "candidate_conformer_status": candidate_status,
                "matched_feature_count": 0,
                "parent_feature_count": 0,
                "feature_family_recall": None,
                "feature_distance_rmsd": None,
            }

    # MCS 에도 남은 예산을 넘긴다. 시간 초과는 구조 부적합과 다른 상태다.
    atom_map, mcs_timed_out = _mcs_atom_map(
        parent_mol, candidate_mol, timeout_seconds=remaining_budget
    )
    if mcs_timed_out:
        return {
            "status": "budget_exhausted",
            "basis": "etkdg_v3_mmff_ligand_feature_alignment",
            "parent_conformer_count": len(parent_conf_ids),
            "candidate_conformer_count": len(candidate_conf_ids),
            "evaluated_pairs": 0,
            "total_pairs": len(parent_conf_ids) * len(candidate_conf_ids),
            "parent_conformer_status": parent_status,
            "candidate_conformer_status": candidate_status,
            "matched_feature_count": 0,
            "parent_feature_count": 0,
            "feature_family_recall": None,
            "feature_distance_rmsd": None,
        }
    if len(atom_map) < 3:
        return {
            "status": "unavailable",
            "basis": "etkdg_v3_mmff_ligand_feature_alignment",
            "parent_conformer_count": len(parent_conf_ids),
            "candidate_conformer_count": len(candidate_conf_ids),
            "parent_conformer_status": parent_status,
            "candidate_conformer_status": candidate_status,
            "matched_feature_count": 0,
            "parent_feature_count": 0,
            "feature_family_recall": None,
            "feature_distance_rmsd": None,
        }

    cached_parent_features = (
        parent_features_by_conformer
        if parent_features_by_conformer is not None
        else _parent_feature_records_by_conformer(parent_ensemble, factory)
    )
    best: tuple[float, float, int] | None = None
    best_recall = 0.0
    best_rmsd: float | None = None
    best_matched = 0
    best_parent_feature_count = 0
    timed_out = False
    evaluated_pairs = 0
    total_pairs = len(parent_conf_ids) * len(candidate_conf_ids)
    for parent_conf_id in parent_conf_ids:
        if _out_of_time():
            timed_out = True
            break
        parent_features = cached_parent_features.get(int(parent_conf_id), [])
        if not parent_features:
            continue
        for candidate_conf_id in candidate_conf_ids:
            if _out_of_time():
                timed_out = True
                break
            probe = Chem.Mol(candidate_3d)
            try:
                AllChem.AlignMol(
                    probe,
                    parent_3d,
                    prbCid=int(candidate_conf_id),
                    refCid=int(parent_conf_id),
                    atomMap=atom_map,
                )
            except Exception:
                continue
            candidate_features = _feature_records(probe, factory, candidate_conf_id)
            matched, rmsd = _matched_feature_rmsd(parent_features, candidate_features)
            evaluated_pairs += 1
            recall = matched / len(parent_features) if parent_features else 0.0
            rmsd_for_sort = rmsd if rmsd is not None else math.inf
            sort_key = (-recall, rmsd_for_sort, int(parent_conf_id) * 10_000 + int(candidate_conf_id))
            if best is None or sort_key < best:
                best = sort_key
                best_recall = recall
                best_rmsd = rmsd
                best_matched = matched
                best_parent_feature_count = len(parent_features)
    # 정렬이 실제로 돌았는지와 RMSD 를 정의할 수 있는지는 다른 물음이다.
    # RMSD 는 맞은 특징이 3개 미만이면 정의되지 않는데, 예전에는 그때 이미
    # 계산해 둔 회수율까지 함께 버렸다. 그래서 "질의의 파마코포어 특징을 하나도
    # 공유하지 않는다"는, 이 화면이 낼 수 있는 가장 강한 부정 근거가 "아무도
    # 재지 못했다"와 똑같이 표시됐다. 특징 17개짜리 질의에서는 3/17=0.176 미만의
    # 회수율이 화면에 아예 나타날 수 없기도 했다.
    if best is None:
        # 한 쌍도 못 돌아 본 채 시간이 다 됐으면 "정렬할 수 없다"가 아니다.
        status = "budget_exhausted" if timed_out else "unavailable"
    elif timed_out:
        status = "partial_timeout"
    elif best_rmsd is not None:
        status = "available"
    else:
        status = "recall_only"
    return {
        "status": status,
        "basis": "etkdg_v3_mmff_ligand_feature_alignment",
        "parent_conformer_count": len(parent_conf_ids),
        "candidate_conformer_count": len(candidate_conf_ids),
        "evaluated_pairs": evaluated_pairs,
        "total_pairs": total_pairs,
        "parent_conformer_status": parent_status,
        "candidate_conformer_status": candidate_status,
        "matched_feature_count": best_matched,
        "parent_feature_count": best_parent_feature_count,
        "feature_family_recall": (
            _round(best_recall) if status in {"available", "recall_only"} else None
        ),
        "feature_distance_rmsd": _round(best_rmsd) if status == "available" else None,
    }


# 3D 비교가 **정렬 가능한 답**을 내지 못한 상태들. 이유는 서로 다르지만, 하류에서
# "이 후보에는 3D 근거가 없다"로 세는 자리에서는 같이 세야 한다. 예전에는
# `== "unavailable"` 하나만 봤는데, 그 뒤로 상태가 늘면서 집계가 실제보다
# 작게 나오게 됐다.
NO_3D_EVIDENCE_STATUSES = frozenset({
    "unavailable",           # 컨포머 실패 또는 공통 구조 3원자 미만
    "parent_unavailable",    # 입력 분자 쪽 앙상블 실패 - 후보와 무관하다
    "budget_exhausted",      # 시간 상한
    "partial_timeout",       # 일부 쌍만 평가한 뒤 시간 상한; 완전 탐색 점수가 아님
    "recall_only",           # 정렬은 됐지만 맞은 특징이 3개 미만이라 RMSD 가 없다
    "not_evaluated_budget",  # 후보 예산 밖
})


def _assign_pareto_fronts(rows: list[dict[str, Any]]) -> None:
    objectives = (
        "pharmacophore_preservation_score",
        "binding_support_score",
        "safety_triage_score",
        "routeability_proxy",
        "corpus_novelty_proxy",
    )

    def values(row: dict[str, Any]) -> tuple[float, ...]:
        parsed: list[float] = []
        for name in objectives:
            raw_value = row.get(name)
            if isinstance(raw_value, bool) or raw_value is None:
                raise SystemExit(
                    f"Pareto objective {name} must be a finite value in [0, 1]"
                )
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise SystemExit(
                    f"Pareto objective {name} must be a finite value in [0, 1]"
                ) from exc
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise SystemExit(
                    f"Pareto objective {name} must be a finite value in [0, 1]"
                )
            parsed.append(value)
        return tuple(parsed)

    row_values = [values(row) for row in rows]
    dominates: list[set[int]] = [set() for _ in rows]
    dominated_by_count = [0 for _ in rows]
    for left_index, left_values in enumerate(row_values):
        for right_index, right_values in enumerate(row_values):
            if left_index == right_index:
                continue
            left_dominates = all(
                left >= right for left, right in zip(left_values, right_values, strict=True)
            ) and any(left > right for left, right in zip(left_values, right_values, strict=True))
            if left_dominates:
                dominates[left_index].add(right_index)
                dominated_by_count[right_index] += 1

    remaining = set(range(len(rows)))
    front = 1
    counts = dominated_by_count[:]
    while remaining:
        current = sorted(index for index in remaining if counts[index] == 0)
        if not current:
            current = sorted(remaining)
        for index in current:
            rows[index]["pareto_front"] = front
            rows[index]["pareto_dominated_by_count"] = dominated_by_count[index]
            rows[index]["pareto_dominates_count"] = len(dominates[index])
            rows[index]["pareto_objectives"] = {
                name: rows[index].get(name) for name in objectives
            }
            remaining.remove(index)
        for index in current:
            for dominated_index in dominates[index]:
                counts[dominated_index] -= 1
        front += 1


def _specified_stereo_elements(molecule: Chem.Mol) -> int:
    return sum(
        info.specified == Chem.StereoSpecified.Specified
        for info in Chem.FindPotentialStereo(molecule)
    )


def _is_less_specific_same_connectivity(
    parent: MoleculeIdentity,
    candidate: MoleculeIdentity,
) -> bool:
    if parent.inchikey.split("-", 1)[0] != candidate.inchikey.split("-", 1)[0]:
        return False
    return _specified_stereo_elements(candidate.mol) < _specified_stereo_elements(
        parent.mol
    )


def _alert_catalogs() -> dict[str, FilterCatalog.FilterCatalog]:
    catalogs: dict[str, FilterCatalog.FilterCatalog] = {}
    for name, enum_value in ALERT_CATALOGS.items():
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(enum_value)
        catalogs[name] = FilterCatalog.FilterCatalog(params)
    return catalogs


def _structural_alerts(
    molecule: Chem.Mol,
    catalogs: dict[str, FilterCatalog.FilterCatalog],
) -> dict[str, list[str]]:
    return {
        name: sorted({match.GetDescription() for match in catalog.GetMatches(molecule)})
        for name, catalog in catalogs.items()
    }


def _safety_triage_score(
    property_score: float,
    alerts: dict[str, list[str]],
) -> float:
    alert_count = sum(len(values) for values in alerts.values())
    critical_alert = any(
        alerts[name] for name in ("PAINS_A", "PAINS_B", "PAINS_C", "NIH")
    )
    return _clip(
        property_score
        - min(0.60, 0.08 * alert_count)
        - (0.20 if critical_alert else 0.0)
    )


# Shared with the applicability gate so the two cannot drift apart.
_physchem = physchem


def _property_score(values: dict[str, float]) -> float:
    checks = (
        1.0 if 40.0 <= values["molecular_weight"] <= 500.0 else 0.25,
        1.0 if -1.0 <= values["logp"] <= 5.0 else 0.25,
        1.0 if 5.0 <= values["tpsa"] <= 140.0 else 0.4,
        1.0 if values["hbd"] <= 5.0 else 0.4,
        1.0 if values["hba"] <= 10.0 else 0.4,
        1.0 if values["rotatable_bonds"] <= 12.0 else 0.4,
    )
    return sum(checks) / len(checks)


def _routeability_proxy(values: dict[str, float], molecule: Chem.Mol) -> float:
    bridgeheads = rdMolDescriptors.CalcNumBridgeheadAtoms(molecule)
    stereocenters = len(Chem.FindMolChiralCenters(molecule, includeUnassigned=True))
    macrocycles = sum(1 for ring in molecule.GetRingInfo().AtomRings() if len(ring) >= 8)
    complexity_penalty = (
        max(0.0, values["heavy_atoms"] - 35.0) / 45.0
        + 0.10 * values["rings"]
        + 0.12 * bridgeheads
        + 0.05 * stereocenters
        + 0.20 * macrocycles
    )
    return _clip(1.0 - complexity_penalty / 2.0)


def _median_activity(records: list[dict[str, Any]]) -> float | None:
    values = [float(record["pactivity"]) for record in records]
    return statistics.median(values) if values else None


def _activity_strata(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for record in records:
        activity_type = _clean(record.get("activity_type")).upper().replace(" ", "")
        source = _clean(record.get("source"))
        pactivity = _finite(record.get("pactivity"))
        if not activity_type or not source or pactivity is None:
            continue
        grouped.setdefault((activity_type, source), []).append(pactivity)
    return [
        {
            "activity_type": activity_type,
            "source": source,
            "median_pactivity": _round(statistics.median(values)),
            "evidence_count": len(values),
        }
        for (activity_type, source), values in sorted(grouped.items())
    ]


def _binding_assessment(
    candidate_records: list[dict[str, Any]],
    parent_records: list[dict[str, Any]],
    pharmacophore_score: float,
    structural_similarity: float,
    max_potency_loss: float,
) -> dict[str, Any]:
    candidate_activity = _median_activity(candidate_records)
    parent_activity = _median_activity(parent_records)
    candidate_strata = {
        (str(row["activity_type"]), str(row["source"])): row
        for row in _activity_strata(candidate_records)
    }
    parent_strata = {
        (str(row["activity_type"]), str(row["source"])): row
        for row in _activity_strata(parent_records)
    }
    comparison_strata = []
    for activity_type, source in sorted(candidate_strata.keys() & parent_strata.keys()):
        candidate_value = float(candidate_strata[(activity_type, source)]["median_pactivity"])
        parent_value = float(parent_strata[(activity_type, source)]["median_pactivity"])
        comparison_strata.append(
            {
                "activity_type": activity_type,
                "source": source,
                "candidate_median_pactivity": _round(candidate_value),
                "parent_median_pactivity": _round(parent_value),
                "pactivity_delta": _round(candidate_value - parent_value),
                "candidate_evidence_count": candidate_strata[(activity_type, source)][
                    "evidence_count"
                ],
                "parent_evidence_count": parent_strata[(activity_type, source)][
                    "evidence_count"
                ],
            }
        )
    if comparison_strata:
        delta = min(float(row["pactivity_delta"]) for row in comparison_strata)
        retained = delta >= -max_potency_loss
        return {
            "evidence_tier": "direct_retained" if retained else "direct_reduced",
            "status": (
                "same_endpoint_source_activity_retention_supported"
                if retained
                else "same_endpoint_source_potency_reduction_observed"
            ),
            "score": _clip((delta + 2.0) / 2.0),
            "pactivity_delta": delta,
            "retained": retained,
            "comparison_basis": "same_activity_type_and_source_conservative_delta",
            "comparison_strata": comparison_strata,
        }
    if candidate_activity is not None:
        return {
            "evidence_tier": "direct_activity",
            "status": (
                "same_target_activity_supported_no_comparable_parent_stratum"
                if parent_activity is not None
                else "same_target_activity_supported_parent_baseline_missing"
            ),
            "score": 0.45 + 0.55 * _clip((candidate_activity - 4.0) / 6.0),
            "pactivity_delta": None,
            "retained": None,
            "comparison_basis": None,
            "comparison_strata": [],
        }
    return {
        "evidence_tier": "proxy_only",
        "status": "ligand_feature_proxy_only",
        "score": min(0.55, 0.40 * pharmacophore_score + 0.15 * structural_similarity),
        "pactivity_delta": None,
        "retained": None,
        "comparison_basis": None,
        "comparison_strata": [],
    }


def _candidate_priority(
    binding: dict[str, Any],
    cosing_reference: bool,
    alerts: dict[str, list[str]],
) -> str:
    critical_alert = any(alerts[name] for name in ("PAINS_A", "PAINS_B", "PAINS_C", "NIH"))
    if binding["evidence_tier"] == "direct_retained" and cosing_reference and not critical_alert:
        return "priority_validation"
    if binding["evidence_tier"].startswith("direct_") and not critical_alert:
        return "target_evidence_review"
    if cosing_reference and not critical_alert:
        return "cosmetic_proxy_review"
    return "research_hypothesis"


def _selection_tracks(row: dict[str, Any]) -> list[str]:
    tracks: list[str] = []
    if row.get("target_conditioned_strict_gate_passed") is True:
        tracks.append("target_conditioned")
    if row["evidence_tier"] != "proxy_only":
        tracks.append("target_activity")
    if row["cosing_reference"]:
        tracks.append("cosmetic_material")
    if not tracks:
        tracks.append("feature_proxy")
    return tracks


def _select_candidate_basket(
    ordered_rows: list[dict[str, Any]],
    *,
    max_candidates: int,
    material_track_fraction: float,
    pharmacophore_track_fraction: float = DEFAULT_PHARMACOPHORE_TRACK_FRACTION,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    for global_rank, row in enumerate(ordered_rows, start=1):
        row["global_priority_rank"] = global_rank
        row["selection_tracks"] = _selection_tracks(row)

    target_rows = [
        row for row in ordered_rows if "target_activity" in row["selection_tracks"]
    ]
    material_rows = [
        row for row in ordered_rows if "cosmetic_material" in row["selection_tracks"]
    ]
    pharmacophore_rows = [
        row for row in ordered_rows if row.get("pharmacophore_gate_passed") is True
    ]
    use_balanced_tracks = bool(
        max_candidates >= 2
        and material_track_fraction > 0.0
        and target_rows
        and material_rows
    )
    reserved_material_slots = (
        min(
            max_candidates,
            len(material_rows),
            max(1, math.ceil(max_candidates * material_track_fraction)),
        )
        if use_balanced_tracks
        else 0
    )
    reserved_pharmacophore_slots = (
        min(
            max_candidates,
            len(pharmacophore_rows),
            max(1, math.ceil(max_candidates * pharmacophore_track_fraction)),
        )
        if max_candidates >= 2
        and pharmacophore_track_fraction > 0.0
        and pharmacophore_rows
        else 0
    )

    selected: list[dict[str, Any]] = []
    selected_smiles: set[str] = set()

    def add(rows: Iterable[dict[str, Any]], limit: int) -> None:
        if limit <= 0 or len(selected) >= max_candidates:
            return
        added = 0
        for row in rows:
            smiles = str(row["smiles"])
            if smiles in selected_smiles:
                continue
            selected.append(row)
            selected_smiles.add(smiles)
            added += 1
            if added >= limit or len(selected) >= max_candidates:
                return

    if use_balanced_tracks:
        add(pharmacophore_rows, reserved_pharmacophore_slots)
        add(material_rows, reserved_material_slots)
        add(target_rows, max_candidates - len(selected))
    elif reserved_pharmacophore_slots:
        add(pharmacophore_rows, reserved_pharmacophore_slots)
    add(ordered_rows, max_candidates - len(selected))
    selected.sort(
        key=lambda row: (
            -float(row["priority_score"]),
            int(row.get("pareto_front") or 999_999),
            int(row["global_priority_rank"]),
            str(row["smiles"]),
        )
    )
    selected_target = sum(
        "target_activity" in row["selection_tracks"] for row in selected
    )
    selected_material = sum(
        "cosmetic_material" in row["selection_tracks"] for row in selected
    )
    selected_pharmacophore = sum(
        row.get("pharmacophore_gate_passed") is True for row in selected
    )
    selected_3d_pharmacophore = sum(
        row.get("strict_3d_pharmacophore_gate_passed") is True for row in selected
    )
    selected_target_conditioned = sum(
        row.get("target_conditioned_strict_gate_passed") is True for row in selected
    )
    strategy = {
        "mode": (
            "balanced_tracks"
            if use_balanced_tracks or reserved_pharmacophore_slots
            else "global_priority"
        ),
        "material_track_fraction": material_track_fraction,
        "pharmacophore_track_fraction": pharmacophore_track_fraction,
        "reserved_material_slots": reserved_material_slots,
        "reserved_pharmacophore_slots": reserved_pharmacophore_slots,
        "eligible_counts": {
            "all": len(ordered_rows),
            "target_activity": len(target_rows),
            "cosmetic_material": len(material_rows),
            "strict_pharmacophore": len(pharmacophore_rows),
        },
        "selected_counts": {
            "all": len(selected),
            "target_activity": selected_target,
            "cosmetic_material": selected_material,
            "strict_pharmacophore": selected_pharmacophore,
            "strict_3d_pharmacophore": selected_3d_pharmacophore,
            "strict_target_conditioned": selected_target_conditioned,
            "both": sum(len(row["selection_tracks"]) > 1 for row in selected),
        },
    }
    return selected, strategy


def _three_d_evaluation_priority(
    ordered_rows: list[dict[str, Any]],
    *,
    max_candidates: int,
    material_track_fraction: float,
    pharmacophore_track_fraction: float,
) -> list[dict[str, Any]]:
    """Prioritize the cheap-score basket, then stable global-score overflow."""
    basket_rows, _strategy = _select_candidate_basket(
        ordered_rows,
        max_candidates=max_candidates,
        material_track_fraction=material_track_fraction,
        pharmacophore_track_fraction=pharmacophore_track_fraction,
    )
    basket_smiles = {str(row["smiles"]) for row in basket_rows}
    return basket_rows + [
        row for row in ordered_rows if str(row["smiles"]) not in basket_smiles
    ]


def _budget_excluded_3d_result(
    parent_ensemble: tuple[Chem.Mol | None, list[int], str],
) -> dict[str, Any]:
    _parent_3d, parent_conf_ids, parent_status = parent_ensemble
    return {
        "status": "not_evaluated_budget",
        "basis": "etkdg_v3_mmff_ligand_feature_alignment",
        "parent_conformer_count": len(parent_conf_ids),
        "candidate_conformer_count": 0,
        "parent_conformer_status": parent_status,
        "candidate_conformer_status": "not_evaluated_budget",
        "matched_feature_count": None,
        "parent_feature_count": None,
        "feature_family_recall": None,
        "feature_distance_rmsd": None,
    }


def _apply_pharmacophore_3d_result(
    row: dict[str, Any],
    result: dict[str, Any],
    *,
    evaluated: bool,
    evaluation_rank: int | None,
    min_3d_feature_recall: float,
    max_feature_distance_rmsd: float,
) -> None:
    feature_recall = result["feature_family_recall"]
    feature_distance_rmsd = result["feature_distance_rmsd"]
    strict_3d_gate_passed = bool(
        evaluated
        and result["status"] == "available"
        and feature_recall is not None
        and feature_distance_rmsd is not None
        and float(feature_recall) >= min_3d_feature_recall
        and float(feature_distance_rmsd) <= max_feature_distance_rmsd
    )
    anchor_score = row.get("target_conditioned_anchor_score")
    target_conditioned_strict_gate_passed = (
        bool(
            row.get("target_conditioned_anchor_gate_passed") is True
            and strict_3d_gate_passed
        )
        if anchor_score is not None
        else None
    )
    target_activity_admitted = (
        "same_target_activity_feature_family" in row["admission_bases"]
    )

    row.update(
        {
            "pharmacophore_3d_status": result["status"],
            "pharmacophore_3d_evaluated_pairs": int(result.get("evaluated_pairs") or 0),
            "pharmacophore_3d_total_pairs": int(result.get("total_pairs") or 0),
            "pharmacophore_3d_basis": result["basis"],
            "pharmacophore_3d_parent_conformer_count": result[
                "parent_conformer_count"
            ],
            "pharmacophore_3d_candidate_conformer_count": result[
                "candidate_conformer_count"
            ],
            "pharmacophore_3d_parent_conformer_status": result[
                "parent_conformer_status"
            ],
            "pharmacophore_3d_candidate_conformer_status": result[
                "candidate_conformer_status"
            ],
            "pharmacophore_3d_matched_feature_count": result[
                "matched_feature_count"
            ],
            "pharmacophore_3d_parent_feature_count": result[
                "parent_feature_count"
            ],
            "pharmacophore_3d_feature_family_recall": feature_recall,
            "pharmacophore_3d_feature_distance_rmsd": feature_distance_rmsd,
            "strict_3d_pharmacophore_gate_passed": strict_3d_gate_passed,
            "target_conditioned_strict_gate_passed": (
                target_conditioned_strict_gate_passed
            ),
            "pharmacophore_3d_evaluated": evaluated,
            "pharmacophore_3d_evaluation_rank": evaluation_rank,
        }
    )
    row["admission_bases"] = [
        basis
        for basis, admitted in (
            ("strict_2d_pharmacophore", row["pharmacophore_gate_passed"] is True),
            ("strict_3d_pharmacophore", strict_3d_gate_passed),
            (
                "strict_target_conditioned_pharmacophore",
                target_conditioned_strict_gate_passed is True,
            ),
            ("same_target_activity_feature_family", target_activity_admitted),
        )
        if admitted
    ]


def _score_candidates(
    parent: MoleculeIdentity,
    seeds: dict[str, CandidateSeed],
    *,
    min_pharmacophore: float,
    min_feature_recall: float,
    max_structural_similarity: float,
    max_potency_loss: float,
    max_candidates: int,
    material_track_fraction: float,
    pharmacophore_track_fraction: float = DEFAULT_PHARMACOPHORE_TRACK_FRACTION,
    interaction_anchor_payload: dict[str, Any] | None = None,
    target_id: str | None = None,
    seed: int = 49242,
    max_3d_conformers: int = DEFAULT_MAX_3D_CONFORMERS,
    max_3d_candidates: int = DEFAULT_MAX_3D_CANDIDATES,
    min_3d_feature_recall: float = DEFAULT_MIN_3D_FEATURE_RECALL,
    max_feature_distance_rmsd: float = DEFAULT_MAX_FEATURE_DISTANCE_RMSD,
    min_anchor_preservation: float = DEFAULT_MIN_ANCHOR_PRESERVATION,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    feature_factory = ChemicalFeatures.BuildFeatureFactory(
        str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef")
    )
    morgan = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
        includeChirality=True,
    )
    parent_pharm = Generate.Gen2DFingerprint(parent.mol, Gobbi_Pharm2D.factory)
    parent_features = _feature_counts(parent.mol, feature_factory)
    if not parent_features:
        raise SystemExit(
            "Input compound has no supported ligand-feature pharmacophore; "
            "substitute discovery cannot make a preservation comparison"
        )
    parent_pharm_usable = parent_pharm.GetNumOnBits() > 0
    parent_fp = morgan.GetFingerprint(parent.mol)
    catalogs = _alert_catalogs()
    parent_alerts = _structural_alerts(parent.mol, catalogs)
    parent_properties = _physchem(parent.mol)
    parent_property_score = _property_score(parent_properties)
    parent_safety_score = _safety_triage_score(
        parent_property_score,
        parent_alerts,
    )
    parent_seed = seeds.get(parent.smiles)
    parent_activities = parent_seed.activities if parent_seed else []
    parent_activity = _median_activity(parent_activities)
    anchor_context = _target_anchor_context(
        parent,
        interaction_anchor_payload,
        target_id,
    )

    rows: list[dict[str, Any]] = []
    candidate_molecules: dict[str, Chem.Mol] = {}
    candidate_3d_seeds: dict[str, int] = {}
    exclusions = Counter(
        parent_identity=0,
        invalid=0,
        pharmacophore=0,
        feature_recall=0,
        stereo_underspecified=0,
        near_duplicate=0,
    )
    for canonical in sorted(seeds):
        candidate_seed = seeds[canonical]
        if canonical == parent.smiles:
            exclusions["parent_identity"] += 1
            continue
        molecule = Chem.MolFromSmiles(canonical)
        if molecule is None:
            exclusions["invalid"] += 1
            continue
        candidate_pharm = Generate.Gen2DFingerprint(molecule, Gobbi_Pharm2D.factory)
        candidate_features = _feature_counts(molecule, feature_factory)
        feature_recall, feature_precision, feature_f1 = _feature_overlap_scores(
            parent_features, candidate_features
        )
        candidate_pharm_usable = candidate_pharm.GetNumOnBits() > 0
        pharmacophore_similarity = (
            float(DataStructs.TanimotoSimilarity(parent_pharm, candidate_pharm))
            if parent_pharm_usable and candidate_pharm_usable
            else feature_f1
        )
        pharmacophore_score = 0.65 * pharmacophore_similarity + 0.35 * feature_recall
        pharmacophore_gate_passed = (
            pharmacophore_score >= min_pharmacophore
            and feature_recall >= min_feature_recall
        )
        target_activity_feature_analogue = (
            bool(candidate_seed.activities) and feature_recall >= min_feature_recall
        )
        if not pharmacophore_gate_passed and not target_activity_feature_analogue:
            exclusion = (
                "feature_recall"
                if feature_recall < min_feature_recall
                else "pharmacophore"
            )
            exclusions[exclusion] += 1
            continue
        candidate_identity = MoleculeIdentity(
            mol=molecule,
            smiles=canonical,
            inchikey=candidate_seed.inchikey,
        )
        if _is_less_specific_same_connectivity(parent, candidate_identity):
            exclusions["stereo_underspecified"] += 1
            continue
        structural_similarity = float(
            DataStructs.TanimotoSimilarity(parent_fp, morgan.GetFingerprint(molecule))
        )
        if structural_similarity >= max_structural_similarity:
            exclusions["near_duplicate"] += 1
            continue
        anchor_result = (
            score_anchor_preservation(
                anchor_context.parent_mol,
                molecule,
                anchor_context.parent_atom_indices,
                feature_factory,
            )
            if anchor_context is not None
            else None
        )
        properties = _physchem(molecule)
        property_score = _property_score(properties)
        alerts = _structural_alerts(molecule, catalogs)
        alert_count = sum(len(values) for values in alerts.values())
        critical_alert = any(alerts[name] for name in ("PAINS_A", "PAINS_B", "PAINS_C", "NIH"))
        safety_score = _safety_triage_score(property_score, alerts)
        safety_delta = safety_score - parent_safety_score
        routeability = _routeability_proxy(properties, molecule)
        candidate_activity = _median_activity(candidate_seed.activities)
        binding = _binding_assessment(
            candidate_seed.activities,
            parent_activities,
            pharmacophore_score,
            structural_similarity,
            max_potency_loss,
        )
        novelty = 1.0 - structural_similarity
        material_score = 1.0 if candidate_seed.cosing_reference else 0.35
        priority_score = (
            (0.20 if anchor_result is not None else 0.30) * pharmacophore_score
            + (0.10 * anchor_result.score if anchor_result is not None else 0.0)
            + 0.25 * float(binding["score"])
            + 0.15 * safety_score
            + 0.10 * routeability
            + 0.10 * material_score
            + 0.10 * novelty
        )
        evidence_sources = sorted({record["source"] for record in candidate_seed.activities})
        citations = sorted(
            (
                {
                    "source": record["source"],
                    "release": record["release"],
                    "doi": record["doi"],
                    "pmid": record["pmid"],
                }
                for record in candidate_seed.activities
                if record["doi"] or record["pmid"]
            ),
            key=lambda item: (item["source"], item["doi"], item["pmid"]),
        )
        deduped_citations: list[dict[str, str]] = []
        seen_citations: set[tuple[str, str, str]] = set()
        for citation in citations:
            key = (citation["source"], citation["doi"], citation["pmid"])
            if key not in seen_citations:
                seen_citations.add(key)
                deduped_citations.append(citation)
            if len(deduped_citations) == 5:
                break
        candidate_molecules[canonical] = molecule
        candidate_3d_seeds[canonical] = seed + len(rows) + 1
        rows.append(
            {
                "candidate_id": "",
                "name": (
                    candidate_seed.preferred_name
                    or (sorted(candidate_seed.names)[0] if candidate_seed.names else "unnamed activity ligand")
                ),
                "alternate_names": [
                    name
                    for name in sorted(candidate_seed.names)
                    if name != candidate_seed.preferred_name
                ][:5],
                "smiles": canonical,
                "inchikey": candidate_seed.inchikey,
                "candidate_sources": sorted(candidate_seed.sources),
                "cosing_reference": candidate_seed.cosing_reference,
                "cosmetic_functions": sorted(candidate_seed.functions),
                "admission_bases": [
                    basis
                    for basis, admitted in (
                        ("strict_2d_pharmacophore", pharmacophore_gate_passed),
                        (
                            "same_target_activity_feature_family",
                            target_activity_feature_analogue,
                        ),
                    )
                    if admitted
                ],
                "pharmacophore_gate_passed": pharmacophore_gate_passed,
                "pharmacophore_similarity": _round(pharmacophore_similarity),
                "pharmacophore_fingerprint_usable": (
                    parent_pharm_usable and candidate_pharm_usable
                ),
                "feature_family_recall": _round(feature_recall),
                "feature_family_precision": _round(feature_precision),
                "feature_family_f1": _round(feature_f1),
                "pharmacophore_preservation_score": _round(pharmacophore_score),
                "pharmacophore_3d_status": "not_evaluated_budget",
                "pharmacophore_3d_basis": (
                    "etkdg_v3_mmff_ligand_feature_alignment"
                ),
                "pharmacophore_3d_parent_conformer_count": 0,
                "pharmacophore_3d_candidate_conformer_count": 0,
                "pharmacophore_3d_parent_conformer_status": "not_evaluated",
                "pharmacophore_3d_candidate_conformer_status": (
                    "not_evaluated_budget"
                ),
                "pharmacophore_3d_matched_feature_count": None,
                "pharmacophore_3d_parent_feature_count": None,
                "pharmacophore_3d_feature_family_recall": None,
                "pharmacophore_3d_feature_distance_rmsd": None,
                "strict_3d_pharmacophore_gate_passed": False,
                "target_conditioned_anchor_score": (
                    _round(anchor_result.score) if anchor_result is not None else None
                ),
                "target_conditioned_anchor_gate_passed": (
                    anchor_result.score >= min_anchor_preservation
                    if anchor_result is not None
                    else None
                ),
                "target_conditioned_strict_gate_passed": (
                    False if anchor_result is not None else None
                ),
                "target_conditioned_anchor_count": (
                    anchor_result.anchor_count if anchor_result is not None else None
                ),
                "target_conditioned_preserved_anchor_count": (
                    anchor_result.preserved_anchor_count
                    if anchor_result is not None
                    else None
                ),
                "target_conditioned_anchor_basis": (
                    anchor_result.basis if anchor_result is not None else None
                ),
                "target_conditioned_mapping_count": (
                    anchor_result.mapping_count if anchor_result is not None else None
                ),
                "target_conditioned_mapping_ambiguous": (
                    anchor_result.mapping_ambiguous
                    if anchor_result is not None
                    else None
                ),
                "target_conditioned_mapping_truncated": (
                    anchor_result.mapping_truncated
                    if anchor_result is not None
                    else None
                ),
                "analog_pose_verified": False,
                "structural_similarity_to_parent": _round(structural_similarity),
                "stereo_specified_elements": _specified_stereo_elements(molecule),
                "corpus_novelty_proxy": _round(novelty),
                "target_activity_median_pactivity": _round(candidate_activity),
                "parent_activity_median_pactivity": _round(parent_activity),
                "binding_pactivity_delta": _round(binding["pactivity_delta"]),
                "binding_support_score": _round(float(binding["score"])),
                "binding_status": binding["status"],
                "binding_retained": binding["retained"],
                "binding_comparison_basis": binding["comparison_basis"],
                "binding_comparison_strata": binding["comparison_strata"],
                "evidence_tier": binding["evidence_tier"],
                "activity_evidence_count": len(candidate_seed.activities),
                "activity_sources": evidence_sources,
                "activity_strata": _activity_strata(candidate_seed.activities),
                "activity_citations": deduped_citations,
                "structural_alerts": alerts,
                "structural_alert_count": alert_count,
                "property_suitability_score": _round(property_score),
                "safety_triage_score": _round(safety_score),
                "safety_score_delta_vs_parent": _round(safety_delta),
                "routeability_proxy": _round(routeability),
                "physicochemical": {key: _round(value) for key, value in properties.items()},
                "priority_score": _round(priority_score),
                "priority_label": _candidate_priority(binding, candidate_seed.cosing_reference, alerts),
                "claimable": False,
                "hypothesis_only": True,
                "wet_lab_required": True,
                "pharmacophore_3d_evaluated": False,
                "pharmacophore_3d_evaluation_rank": None,
            }
        )

    tier_order = {
        "direct_retained": 0,
        "direct_activity": 1,
        "direct_reduced": 2,
        "proxy_only": 3,
    }
    _assign_pareto_fronts(rows)
    rows.sort(
        key=lambda row: (
            -float(row["priority_score"]),
            int(row.get("pareto_front") or 999_999),
            tier_order[str(row["evidence_tier"])],
            str(row["smiles"]),
        )
    )
    passing_count = len(rows)
    three_d_priority = _three_d_evaluation_priority(
        rows,
        max_candidates=max_candidates,
        material_track_fraction=material_track_fraction,
        pharmacophore_track_fraction=pharmacophore_track_fraction,
    )
    parent_3d_ensemble = _conformer_ensemble(
        parent.mol,
        seed=seed,
        max_conformers=max_3d_conformers,
    )
    parent_features_by_conformer = _parent_feature_records_by_conformer(
        parent_3d_ensemble,
        feature_factory,
    )
    budget_excluded_result = _budget_excluded_3d_result(parent_3d_ensemble)
    for row in rows:
        _apply_pharmacophore_3d_result(
            row,
            budget_excluded_result,
            evaluated=False,
            evaluation_rank=None,
            min_3d_feature_recall=min_3d_feature_recall,
            max_feature_distance_rmsd=max_feature_distance_rmsd,
        )

    evaluated_rows = three_d_priority[:max_3d_candidates]
    for evaluation_rank, row in enumerate(evaluated_rows, start=1):
        canonical = str(row["smiles"])
        pharmacophore_3d = _pharmacophore_3d_comparison(
            parent.mol,
            candidate_molecules[canonical],
            feature_factory,
            parent_3d_ensemble,
            seed=candidate_3d_seeds[canonical],
            max_conformers=max_3d_conformers,
            parent_features_by_conformer=parent_features_by_conformer,
        )
        _apply_pharmacophore_3d_result(
            row,
            pharmacophore_3d,
            evaluated=True,
            evaluation_rank=evaluation_rank,
            min_3d_feature_recall=min_3d_feature_recall,
            max_feature_distance_rmsd=max_feature_distance_rmsd,
        )

    rows, selection_strategy = _select_candidate_basket(
        rows,
        max_candidates=max_candidates,
        material_track_fraction=material_track_fraction,
        pharmacophore_track_fraction=pharmacophore_track_fraction,
    )
    for rank, row in enumerate(rows, start=1):
        row["candidate_id"] = f"SUB{rank:04d}"
        row["rank"] = rank
    parent_summary = {
        "smiles": parent.smiles,
        "inchikey": parent.inchikey,
        "stereo_specified_elements": _specified_stereo_elements(parent.mol),
        "feature_families": dict(sorted(parent_features.items())),
        "pharmacophore_fingerprint_usable": parent_pharm_usable,
        "pharmacophore_3d_conformer_count": len(parent_3d_ensemble[1]),
        "pharmacophore_3d_conformer_status": parent_3d_ensemble[2],
        "activity_median_pactivity": _round(parent_activity),
        "activity_evidence_count": len(parent_activities),
        "activity_sources": sorted({record["source"] for record in parent_activities}),
        "activity_strata": _activity_strata(parent_activities),
        "structural_alerts": parent_alerts,
        "property_suitability_score": _round(parent_property_score),
        "physicochemical": {key: _round(value) for key, value in parent_properties.items()},
        "target_conditioned_interaction_anchor": (
            {
                "enabled": True,
                "target_id": anchor_context.target_id,
                "anchor_count": len(anchor_context.parent_atom_indices),
                "basis": PRESERVATION_BASIS,
                "analog_pose_verified": False,
            }
            if anchor_context is not None
            else {
                "enabled": False,
                "target_id": target_id,
                "anchor_count": 0,
                "basis": None,
                "analog_pose_verified": False,
            }
        ),
    }
    return rows, {
        "parent": parent_summary,
        "exclusions": dict(exclusions),
        "passing_count": passing_count,
        "selection_strategy": selection_strategy,
        "pharmacophore_3d_evaluation": {
            "budget": max_3d_candidates,
            "eligible_candidates": passing_count,
            "evaluated_candidates": len(evaluated_rows),
            "budget_excluded_candidates": passing_count - len(evaluated_rows),
            "available_evidence_candidates": sum(
                row["pharmacophore_3d_status"] == "available"
                for row in evaluated_rows
            ),
            "selection_priority": (
                "existing deterministic 2D/target/cosmetic basket first, then "
                "remaining candidates by legacy global priority, Pareto front, "
                "evidence tier, and canonical SMILES"
            ),
        },
    }


def _csv_value(value: object) -> object:
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "rank",
        "candidate_id",
        "name",
        "smiles",
        "inchikey",
        "candidate_sources",
        "cosing_reference",
        "selection_tracks",
        "global_priority_rank",
        "admission_bases",
        "pharmacophore_gate_passed",
        "pharmacophore_similarity",
        "feature_family_recall",
        "feature_family_precision",
        "feature_family_f1",
        "pharmacophore_preservation_score",
        "pharmacophore_3d_status",
        "pharmacophore_3d_evaluated_pairs",
        "pharmacophore_3d_total_pairs",
        "pharmacophore_3d_basis",
        "pharmacophore_3d_parent_conformer_count",
        "pharmacophore_3d_candidate_conformer_count",
        "pharmacophore_3d_parent_conformer_status",
        "pharmacophore_3d_candidate_conformer_status",
        "pharmacophore_3d_matched_feature_count",
        "pharmacophore_3d_parent_feature_count",
        "pharmacophore_3d_feature_family_recall",
        "pharmacophore_3d_feature_distance_rmsd",
        "strict_3d_pharmacophore_gate_passed",
        "target_conditioned_anchor_score",
        "target_conditioned_anchor_gate_passed",
        "target_conditioned_strict_gate_passed",
        "target_conditioned_anchor_count",
        "target_conditioned_preserved_anchor_count",
        "target_conditioned_anchor_basis",
        "target_conditioned_mapping_count",
        "target_conditioned_mapping_ambiguous",
        "target_conditioned_mapping_truncated",
        "analog_pose_verified",
        "structural_similarity_to_parent",
        "stereo_specified_elements",
        "corpus_novelty_proxy",
        "target_activity_median_pactivity",
        "parent_activity_median_pactivity",
        "binding_pactivity_delta",
        "binding_support_score",
        "binding_status",
        "binding_retained",
        "binding_comparison_basis",
        "binding_comparison_strata",
        "evidence_tier",
        "activity_evidence_count",
        "activity_sources",
        "structural_alert_count",
        "property_suitability_score",
        "safety_triage_score",
        "safety_score_delta_vs_parent",
        "routeability_proxy",
        "pareto_front",
        "pareto_dominated_by_count",
        "pareto_dominates_count",
        "pareto_objectives",
        "priority_score",
        "priority_label",
        "claimable",
        "hypothesis_only",
        "wet_lab_required",
        "pharmacophore_3d_evaluated",
        "pharmacophore_3d_evaluation_rank",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name)) for name in fieldnames})
    tmp.replace(path)


def _conformer(molecule: Chem.Mol, seed: int) -> tuple[Chem.Mol, str]:
    working = Chem.AddHs(Chem.Mol(molecule))
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.useRandomCoords = False
    status = AllChem.EmbedMolecule(working, params)
    if status == 0:
        try:
            AllChem.UFFOptimizeMolecule(working, maxIters=250)
        except Exception:
            pass
        return working, "etkdg_v3"
    fallback = Chem.Mol(molecule)
    AllChem.Compute2DCoords(fallback)
    return fallback, "2d_fallback"


def _write_sdf(
    path: Path,
    parent: dict[str, Any],
    rows: list[dict[str, Any]],
    seed: int,
    limit: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    writer = Chem.SDWriter(str(tmp))
    parent_mol = Chem.MolFromSmiles(str(parent["smiles"]))
    if parent_mol is None:
        writer.close()
        tmp.unlink(missing_ok=True)
        raise SystemExit("Standardized parent molecule could not be serialized")
    parent_3d, status = _conformer(parent_mol, seed)
    parent_3d.SetProp("candidate_id", "PARENT")
    parent_3d.SetProp("role", "parent")
    parent_3d.SetProp("conformer_status", status)
    writer.write(parent_3d)
    # 예산(--max-3d-candidates)만큼 쓴다. 이전에는 128 예산에서도 50개로 잘려
    # 리포트(전체 후보)와 SDF(일부)가 어긋났다.
    for index, row in enumerate(rows[:limit], start=1):
        molecule = Chem.MolFromSmiles(str(row["smiles"]))
        if molecule is None:
            continue
        conformer, conformer_status = _conformer(molecule, seed + index)
        for key in (
            "candidate_id",
            "name",
            "evidence_tier",
            "binding_status",
            "priority_label",
        ):
            conformer.SetProp(key, str(row[key]))
        conformer.SetProp(
            "admission_bases", ",".join(str(item) for item in row["admission_bases"])
        )
        conformer.SetProp(
            "pharmacophore_gate_passed",
            "true" if row["pharmacophore_gate_passed"] else "false",
        )
        conformer.SetProp(
            "selection_tracks", ",".join(str(item) for item in row["selection_tracks"])
        )
        conformer.SetProp("global_priority_rank", str(row["global_priority_rank"]))
        for key in (
            "pharmacophore_preservation_score",
            "pharmacophore_3d_feature_family_recall",
            "pharmacophore_3d_feature_distance_rmsd",
            "target_conditioned_anchor_score",
            "binding_support_score",
            "safety_triage_score",
            "routeability_proxy",
            "pareto_front",
            "priority_score",
        ):
            if row.get(key) is not None:
                conformer.SetProp(key, str(row[key]))
        for key in (
            "pharmacophore_3d_status",
            "pharmacophore_3d_basis",
            "strict_3d_pharmacophore_gate_passed",
            "target_conditioned_anchor_gate_passed",
            "target_conditioned_strict_gate_passed",
        ):
            if row.get(key) is not None:
                value = row[key]
                conformer.SetProp(
                    key,
                    (
                        "true"
                        if value is True
                        else "false" if value is False else str(value)
                    ),
                )
        conformer.SetProp(
            "pharmacophore_3d_evaluated",
            "true" if row["pharmacophore_3d_evaluated"] else "false",
        )
        if row.get("pharmacophore_3d_evaluation_rank") is not None:
            conformer.SetProp(
                "pharmacophore_3d_evaluation_rank",
                str(row["pharmacophore_3d_evaluation_rank"]),
            )
        if row.get("target_conditioned_anchor_score") is not None:
            conformer.SetProp(
                "target_conditioned_anchor_count",
                str(row["target_conditioned_anchor_count"]),
            )
            conformer.SetProp(
                "target_conditioned_preserved_anchor_count",
                str(row["target_conditioned_preserved_anchor_count"]),
            )
            conformer.SetProp(
                "target_conditioned_anchor_basis",
                str(row["target_conditioned_anchor_basis"]),
            )
            conformer.SetProp(
                "target_conditioned_mapping_count",
                str(row["target_conditioned_mapping_count"]),
            )
            conformer.SetProp(
                "target_conditioned_mapping_ambiguous",
                "true" if row["target_conditioned_mapping_ambiguous"] else "false",
            )
            conformer.SetProp(
                "target_conditioned_mapping_truncated",
                "true" if row["target_conditioned_mapping_truncated"] else "false",
            )
        conformer.SetProp("analog_pose_verified", "false")
        conformer.SetProp("claimable", "false")
        conformer.SetProp("hypothesis_only", "true")
        conformer.SetProp("wet_lab_required", "true")
        conformer.SetProp("role", "substitute_hypothesis")
        conformer.SetProp("conformer_status", conformer_status)
        writer.write(conformer)
    writer.close()
    tmp.replace(path)


def _molecule_svg(smiles: str, legend: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return ""
    drawer = rdMolDraw2D.MolDraw2DSVG(260, 170)
    options = drawer.drawOptions()
    options.clearBackground = False
    options.addStereoAnnotation = True
    drawer.DrawMolecule(molecule, legend=legend)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg[svg.find("<svg") :]


def _tier_label(value: str) -> str:
    return {
        "direct_retained": "Matched activity: within threshold",
        "direct_activity": "Direct target activity",
        "direct_reduced": "Matched activity: reduced",
        "proxy_only": "Ligand-feature proxy",
    }.get(value, value)


def _write_html(path: Path, report: dict[str, Any]) -> None:
    target = html.escape(str(report["target"]["target_id"]))
    parent = report["parent"]
    cards: list[str] = []
    for row in report["candidates"]:
        source = "CosIng" if row["cosing_reference"] else "Bioactivity reference"
        raw_binding_delta = row.get("binding_pactivity_delta")
        binding_delta = (
            float(raw_binding_delta)
            if isinstance(raw_binding_delta, (int, float))
            and not isinstance(raw_binding_delta, bool)
            and math.isfinite(float(raw_binding_delta))
            else None
        )
        binding_delta_text = (
            f"{binding_delta:+.3f}" if binding_delta is not None else "Not comparable"
        )
        pharmacophore_status = (
            "Strict 2D gate passed"
            if row["pharmacophore_gate_passed"]
            else "Target-evidence admission; 2D gate failed"
        )
        feature_3d_recall = row.get("pharmacophore_3d_feature_family_recall")
        feature_3d_rmsd = row.get("pharmacophore_3d_feature_distance_rmsd")
        if (
            isinstance(feature_3d_recall, (int, float))
            and not isinstance(feature_3d_recall, bool)
            and isinstance(feature_3d_rmsd, (int, float))
            and not isinstance(feature_3d_rmsd, bool)
        ):
            pharmacophore_3d_text = (
                f"3D recall {float(feature_3d_recall):.3f}; "
                f"RMSD {float(feature_3d_rmsd):.3f} A"
            )
        elif row.get("pharmacophore_3d_status") == "not_evaluated_budget":
            pharmacophore_3d_text = "3D not evaluated (budget)"
        else:
            pharmacophore_3d_text = "3D unavailable"
        anchor_score = row.get("target_conditioned_anchor_score")
        mapping_note = (
            "Conservative minimum across ambiguous mappings"
            if row.get("target_conditioned_mapping_ambiguous") is True
            else "Unique MCS mapping"
        )
        anchor_text = (
            f"{float(anchor_score):.3f}<br><small>{mapping_note}; parent-pose proxy</small>"
            if isinstance(anchor_score, (int, float))
            and not isinstance(anchor_score, bool)
            else "Not available"
        )
        search_terms = [
            row.get("name"),
            row.get("smiles"),
            row.get("candidate_id"),
            row.get("inchikey"),
            *(row.get("alternate_names") or []),
        ]
        search_text = " ".join(
            str(value).strip().lower()
            for value in search_terms
            if str(value or "").strip()
        )
        tracks = " + ".join(
            {
                "target_activity": "Target evidence track",
                "target_conditioned": "Strict target-conditioned track",
                "cosmetic_material": "Cosmetic material track",
                "strict_3d_pharmacophore": "Strict 3D pharmacophore track",
                "feature_proxy": "Feature proxy track",
            }.get(str(track), str(track))
            for track in row["selection_tracks"]
        )
        cards.append(
            "<article class='candidate' "
            f"data-tier='{html.escape(str(row['evidence_tier']))}' "
            f"data-source='{'cosing' if row['cosing_reference'] else 'bioactivity'}' "
            f"data-search='{html.escape(search_text)}'>"
            f"<div class='structure'>{_molecule_svg(str(row['smiles']), str(row['candidate_id']))}</div>"
            f"<div class='candidate-body'><div class='candidate-head'><div><span class='rank'>#{row['rank']} / global #{row['global_priority_rank']}</span>"
            f"<h2>{html.escape(str(row['name']))}</h2></div><span class='tier'>{html.escape(_tier_label(str(row['evidence_tier'])))}</span></div>"
            f"<code>{html.escape(str(row['smiles']))}</code>"
            "<dl>"
            f"<div><dt>Pharmacophore</dt><dd>{float(row['pharmacophore_preservation_score']):.3f}<br><small>{html.escape(pharmacophore_status)}; {html.escape(pharmacophore_3d_text)}</small></dd></div>"
            f"<div><dt>Target anchor</dt><dd>{anchor_text}</dd></div>"
            f"<div><dt>Target support</dt><dd>{float(row['binding_support_score']):.3f}</dd></div>"
            f"<div><dt>Pareto front</dt><dd>{int(row['pareto_front'])}</dd></div>"
            f"<div><dt>Matched ΔpActivity</dt><dd>{binding_delta_text}<br><small>Conservative endpoint/source stratum</small></dd></div>"
            f"<div><dt>Activity records</dt><dd>{int(row['activity_evidence_count'])}</dd></div>"
            f"<div><dt>Structure/property triage</dt><dd>{float(row['safety_triage_score']):.3f}</dd></div>"
            f"<div><dt>Routeability proxy</dt><dd>{float(row['routeability_proxy']):.3f}</dd></div>"
            f"<div><dt>Source</dt><dd>{source}</dd></div>"
            f"<div><dt>Track</dt><dd>{html.escape(tracks)}</dd></div>"
            f"<div><dt>Priority</dt><dd>{html.escape(str(row['priority_label']))}</dd></div>"
            "</dl></div></article>"
        )
    limitations = "".join(f"<li>{html.escape(item)}</li>" for item in report["limitations"])
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SkinScout substitute discovery</title><style>
:root{{--ink:#17211d;--muted:#5f6d67;--line:#d7dfda;--paper:#f4f7f5;--accent:#0b7158;--warn:#8b5c12}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:14px/1.5 system-ui,sans-serif}}
header,main{{max-width:1240px;margin:auto;padding:28px}}header{{padding-bottom:14px}}h1{{margin:4px 0 8px;font-size:28px;letter-spacing:0}}
.lede{{max-width:900px;color:var(--muted)}}.notice{{margin:18px 0;padding:14px 16px;border-left:4px solid var(--warn);background:#fff8e9}}
.summary{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:20px 0}}.metric{{padding:14px;border:1px solid var(--line);background:#fff}}
.metric span{{display:block;color:var(--muted);font-size:11px;text-transform:uppercase}}.metric strong{{font-size:18px}}
.controls{{display:flex;gap:10px;flex-wrap:wrap;margin:20px 0}}input,select{{min-height:40px;padding:8px 10px;border:1px solid var(--line);background:#fff}}
input{{flex:1;min-width:220px}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}
.filter-status{{margin:-10px 0 14px;color:var(--muted);font-size:12px}}
.candidate{{display:grid;grid-template-columns:260px 1fr;border:1px solid var(--line);background:#fff}}.structure{{display:grid;place-items:center;border-right:1px solid var(--line)}}
.candidate-body{{min-width:0;padding:16px}}.candidate-head{{display:flex;justify-content:space-between;gap:12px}}h2{{margin:2px 0 10px;font-size:16px;letter-spacing:0}}
.rank,.tier{{color:var(--accent);font-size:11px;font-weight:700}}code{{display:block;overflow-wrap:anywhere;color:var(--muted);font-size:11px}}
dl{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin:14px 0 0}}dl div{{border-top:1px solid var(--line);padding-top:7px}}dt{{color:var(--muted);font-size:10px}}dd{{margin:2px 0;font-weight:700}}
.limitations{{margin-top:28px;padding:18px;border:1px solid var(--line);background:#fff}}[hidden]{{display:none!important}}
@media(max-width:850px){{.summary{{grid-template-columns:repeat(2,minmax(0,1fr))}}.grid{{grid-template-columns:1fr}}.candidate{{grid-template-columns:1fr}}.structure{{border-right:0;border-bottom:1px solid var(--line)}}}}
</style></head><body><header><p class="rank">SKINSCOUT / SUBSTITUTE DISCOVERY</p><h1>Pharmacophore-preserving substitute hypotheses</h1>
<p class="lede">Parent <code>{html.escape(str(parent['smiles']))}</code> was compared for target <strong>{target}</strong>. Scores rank experimental priorities; they do not establish equivalent efficacy or safety.</p></header>
<main><div class="notice"><strong>Hypothesis only.</strong> A matched activity-retention tier requires comparable endpoint/source strata and still needs target-specific assay, formulation, permeation, and safety validation.</div>
<section class="summary"><div class="metric"><span>Candidates</span><strong>{len(report['candidates'])}</strong></div><div class="metric"><span>Direct target evidence</span><strong>{report['summary']['direct_activity_candidates']}</strong></div><div class="metric"><span>Matched activity range</span><strong>{report['summary']['direct_retained_candidates']}</strong></div><div class="metric"><span>CosIng references</span><strong>{report['summary']['cosing_candidates']}</strong></div><div class="metric"><span>3D evaluated / eligible</span><strong>{report['summary']['pharmacophore_3d_evaluated_candidates']} / {report['summary']['pharmacophore_3d_eligible_candidates']}</strong><small>Budget {report['summary']['pharmacophore_3d_candidate_budget']}; excluded {report['summary']['pharmacophore_3d_budget_excluded_candidates']}</small></div></section>
<section class="controls"><input id="query" type="search" placeholder="Search candidate name, ID, InChIKey, or SMILES" aria-label="Search candidates" aria-controls="candidates"><select id="tier" aria-label="Evidence tier" aria-controls="candidates"><option value="all">All evidence tiers</option><option value="direct_retained">Direct retained</option><option value="direct_activity">Direct activity</option><option value="direct_reduced">Direct reduced</option><option value="proxy_only">Proxy only</option></select><select id="source" aria-label="Candidate source" aria-controls="candidates"><option value="all">All sources</option><option value="cosing">CosIng</option><option value="bioactivity">Bioactivity reference</option></select></section>
<p class="filter-status" id="filter-status" aria-live="polite">Showing {len(report['candidates'])} of {len(report['candidates'])} candidates</p>
<section class="grid" id="candidates">{''.join(cards) or '<p>No candidate passed the configured gates.</p>'}</section>
<section class="limitations"><h2>Interpretation limits</h2><ul>{limitations}</ul></section></main>
<script>const q=document.querySelector('#query'),t=document.querySelector('#tier'),s=document.querySelector('#source'),status=document.querySelector('#filter-status'),cards=[...document.querySelectorAll('.candidate')];function filter(){{const text=q.value.trim().toLowerCase();let visible=0;for(const card of cards){{const hidden=Boolean((text&&!card.dataset.search.includes(text))||(t.value!=='all'&&card.dataset.tier!==t.value)||(s.value!=='all'&&card.dataset.source!==s.value));card.hidden=hidden;if(!hidden)visible+=1}}status.textContent=`Showing ${{visible}} of ${{cards.length}} candidates`}}q.addEventListener('input',filter);t.addEventListener('change',filter);s.addEventListener('change',filter);</script></body></html>"""
    _write_text_atomic(path, document)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# SkinScout substitute discovery",
        "",
        f"- Target: `{report['target']['target_id']}`",
        f"- Parent: `{report['parent']['smiles']}`",
        f"- Candidate count: {len(report['candidates'])}",
        (
            "- 3D evaluation: "
            f"{report['summary']['pharmacophore_3d_evaluated_candidates']} / "
            f"{report['summary']['pharmacophore_3d_eligible_candidates']} eligible "
            f"(budget {report['summary']['pharmacophore_3d_candidate_budget']}; "
            f"excluded {report['summary']['pharmacophore_3d_budget_excluded_candidates']})"
        ),
        "- Claim status: hypothesis only; wet-lab validation required",
        "",
        "| Basket rank | Global rank | Pareto front | Candidate | Track | Admission | Pharmacophore | 3D pharmacophore | Target anchor | Target evidence | Structure/property triage | Routeability |",
        "|---:|---:|---:|---|---|---|---:|---|---:|---|---:|---:|",
    ]
    for row in report["candidates"][:30]:
        name = str(row["name"]).replace("|", "\\|")
        tracks = ", ".join(str(track) for track in row["selection_tracks"])
        admission = (
            "strict 2D pharmacophore"
            if row["pharmacophore_gate_passed"]
            else "same-target activity + feature family"
        )
        anchor = row.get("target_conditioned_anchor_score")
        anchor_text = (
            f"{float(anchor):.3f} (conservative)"
            if anchor is not None
            else "n/a"
        )
        feature_3d_recall = row.get("pharmacophore_3d_feature_family_recall")
        feature_3d_rmsd = row.get("pharmacophore_3d_feature_distance_rmsd")
        if feature_3d_recall is not None and feature_3d_rmsd is not None:
            pharmacophore_3d = (
                f"recall {float(feature_3d_recall):.3f}; "
                f"RMSD {float(feature_3d_rmsd):.3f} A"
            )
        elif row.get("pharmacophore_3d_status") == "not_evaluated_budget":
            pharmacophore_3d = "not evaluated (budget)"
        else:
            pharmacophore_3d = "unavailable"
        lines.append(
            f"| {row['rank']} | {row['global_priority_rank']} | {row['pareto_front']} | {name} | {tracks} | {admission} | "
            f"{row['pharmacophore_preservation_score']:.3f} | {pharmacophore_3d} | {anchor_text} | "
            f"{row['evidence_tier']} | {row['safety_triage_score']:.3f} | "
            f"{row['routeability_proxy']:.3f} |"
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    _write_text_atomic(path, "\n".join(lines) + "\n")


def _validate_args(args: argparse.Namespace) -> None:
    if not hasattr(args, "max_3d_candidates"):
        args.max_3d_candidates = DEFAULT_MAX_3D_CANDIDATES
    for value, label in (
        (args.min_pharmacophore, "--min-pharmacophore"),
        (args.min_feature_recall, "--min-feature-recall"),
        (args.max_structural_similarity, "--max-structural-similarity"),
        (args.material_track_fraction, "--material-track-fraction"),
        (args.pharmacophore_track_fraction, "--pharmacophore-track-fraction"),
        (args.min_3d_feature_recall, "--min-3d-feature-recall"),
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise SystemExit(f"{label} must be a finite value in [0, 1]")
    if (
        not math.isfinite(args.max_feature_distance_rmsd)
        or args.max_feature_distance_rmsd < 0.0
    ):
        raise SystemExit("--max-feature-distance-rmsd must be finite and >= 0")
    if (
        not math.isfinite(args.min_anchor_preservation)
        or not 0.0 <= args.min_anchor_preservation <= 1.0
    ):
        raise SystemExit("--min-anchor-preservation must be a finite value in [0, 1]")
    if args.max_potency_loss < 0 or not math.isfinite(args.max_potency_loss):
        raise SystemExit("--max-potency-loss must be finite and >= 0")
    if not 0 <= args.max_3d_conformers <= 50:
        raise SystemExit("--max-3d-conformers must be in [0, 50]")
    if not 1 <= args.max_3d_candidates <= 500:
        raise SystemExit("--max-3d-candidates must be in [1, 500]")
    if not 1 <= args.max_candidates <= 500:
        raise SystemExit("--max-candidates must be in [1, 500]")
    if not 1 <= args.max_direct_molecules <= 100_000:
        raise SystemExit("--max-direct-molecules must be in [1, 100000]")
    target = args.target_id.strip().upper()
    if not UNIPROT_ACCESSION_RE.fullmatch(target):
        raise SystemExit("--target-id must be a valid UniProt accession")
    args.target_id = target


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _require_new_outputs(args.out_dir)
    # 대체 성분 탐색도 도구의 입력 범위 계약을 지킨다(펩타이드·고분자·계면활성제 거부).
    message = refusal_message(assess(args.parent_smiles))
    if message:
        raise SystemExit(message)
    try:
        parent = _standardize_smiles(args.parent_smiles, "Parent compound")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    interaction_anchor_payload: dict[str, Any] | None = None
    if args.interaction_anchor_map is not None:
        try:
            interaction_anchor_payload = load_anchor_map(
                args.interaction_anchor_map,
                expected_parent_inchikey=parent.inchikey,
                required_target=args.target_id,
                verify_source_files=True,
            )
        except InteractionAnchorError as exc:
            raise SystemExit(str(exc)) from exc
    library_seeds, library_counts, library_fingerprint = _candidate_library(
        args.candidate_library
    )
    activity_paths = args.activity_evidence or list(DEFAULT_ACTIVITY_EVIDENCE)
    activity_seeds, activity_audits, activity_summary = _target_activity_candidates(
        activity_paths,
        args.target_id,
        parent,
        max_direct_molecules=args.max_direct_molecules,
    )
    seeds = dict(library_seeds)
    for canonical, activity_seed in activity_seeds.items():
        if canonical in seeds:
            _merge_seed(seeds[canonical], activity_seed)
        else:
            seeds[canonical] = activity_seed
    alias_audits = _enrich_seed_names(seeds, list(args.alias_evidence))
    try:
        rows, scoring = _score_candidates(
            parent,
            seeds,
            min_pharmacophore=args.min_pharmacophore,
            min_feature_recall=args.min_feature_recall,
            max_structural_similarity=args.max_structural_similarity,
            max_potency_loss=args.max_potency_loss,
            max_candidates=args.max_candidates,
            material_track_fraction=args.material_track_fraction,
            pharmacophore_track_fraction=args.pharmacophore_track_fraction,
            interaction_anchor_payload=interaction_anchor_payload,
            target_id=args.target_id,
            seed=args.seed,
            max_3d_conformers=args.max_3d_conformers,
            max_3d_candidates=args.max_3d_candidates,
            min_3d_feature_recall=args.min_3d_feature_recall,
            max_feature_distance_rmsd=args.max_feature_distance_rmsd,
            min_anchor_preservation=args.min_anchor_preservation,
        )
    except InteractionAnchorError as exc:
        raise SystemExit(str(exc)) from exc
    anchor_enabled = interaction_anchor_payload is not None
    three_d_evaluation = scoring["pharmacophore_3d_evaluation"]
    summary = {
        "direct_activity_candidates": sum(
            row["evidence_tier"] != "proxy_only" for row in rows
        ),
        "direct_retained_candidates": sum(
            row["evidence_tier"] == "direct_retained" for row in rows
        ),
        "cosing_candidates": sum(row["cosing_reference"] for row in rows),
        "proxy_only_candidates": sum(row["evidence_tier"] == "proxy_only" for row in rows),
        "strict_pharmacophore_candidates": sum(
            row["pharmacophore_gate_passed"] for row in rows
        ),
        "target_supported_feature_analogue_candidates": sum(
            "same_target_activity_feature_family" in row["admission_bases"]
            for row in rows
        ),
        "target_supported_nonpharmacophore_candidates": sum(
            not row["pharmacophore_gate_passed"]
            and "same_target_activity_feature_family" in row["admission_bases"]
            for row in rows
        ),
        "target_conditioned_anchor_candidates": sum(
            row["target_conditioned_anchor_score"] is not None for row in rows
        ),
        "strict_3d_pharmacophore_candidates": sum(
            row["strict_3d_pharmacophore_gate_passed"] for row in rows
        ),
        "strict_target_conditioned_candidates": sum(
            row.get("target_conditioned_strict_gate_passed") is True for row in rows
        ),
        "pharmacophore_3d_unavailable_candidates": sum(
            row["pharmacophore_3d_status"] in NO_3D_EVIDENCE_STATUSES for row in rows
        ),
        "pharmacophore_3d_candidate_budget": three_d_evaluation["budget"],
        "pharmacophore_3d_eligible_candidates": three_d_evaluation[
            "eligible_candidates"
        ],
        "pharmacophore_3d_evaluated_candidates": three_d_evaluation[
            "evaluated_candidates"
        ],
        "pharmacophore_3d_budget_excluded_candidates": three_d_evaluation[
            "budget_excluded_candidates"
        ],
    }
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed" if rows else "completed_no_candidates",
        "generated_at_utc": _utc_now(),
        "run_id": args.run_id,
        "claimable": False,
        "hypothesis_only": True,
        "wet_lab_required": True,
        "parent": scoring["parent"],
        "target": {
            "target_id": args.target_id,
            "selection_basis": args.target_selection_basis,
            "interaction_anchor_conditioned": anchor_enabled,
            "binding_interpretation": (
                "Same-target exact public activity evidence is preferred. "
                "Ligand-feature similarity alone is a non-binding proxy."
            ),
        },
        "thresholds": {
            "min_pharmacophore_preservation_score": args.min_pharmacophore,
            "min_feature_family_recall": args.min_feature_recall,
            "min_3d_feature_family_recall": args.min_3d_feature_recall,
            "max_3d_feature_distance_rmsd_angstrom": args.max_feature_distance_rmsd,
            "max_3d_conformers": args.max_3d_conformers,
            "min_target_conditioned_anchor_preservation": args.min_anchor_preservation,
            "max_structural_similarity_to_parent": args.max_structural_similarity,
            "max_direct_pactivity_loss_for_retention": args.max_potency_loss,
            "max_candidates": args.max_candidates,
            "material_track_fraction": args.material_track_fraction,
            "pharmacophore_track_fraction": args.pharmacophore_track_fraction,
            "max_3d_candidates": args.max_3d_candidates,
        },
        "score_definition": {
            "priority_score": {
                "pharmacophore_preservation": 0.20 if anchor_enabled else 0.30,
                "target_conditioned_interaction_anchor": (
                    0.10 if anchor_enabled else 0.0
                ),
                "binding_support": 0.25,
                "safety_triage": 0.15,
                "routeability_proxy": 0.10,
                "material_reference": 0.10,
                "corpus_novelty_proxy": 0.10,
            },
            "ranking_semantics": (
                "Candidate inclusion balances strict 2D pharmacophore, target-activity, "
                "and cosmetic-material tracks when available. Within the selected basket, "
                "priority_score is the primary ordering key; global_priority_rank preserves "
                "the unstratified score order and canonical SMILES is a deterministic tie-breaker."
            ),
            "pharmacophore_semantics": (
                "RDKit Gobbi 2D ligand-feature fingerprint plus parent feature-family recall; "
                "feature-family F1 is used as the explicit fallback when the pair-based "
                "fingerprint has zero bits, while minimum recall remains a separate gate; "
                "the strict admission gate remains independent of the optional target anchor score"
            ),
            "target_conditioned_anchor_semantics": (
                "Stage 5.5 parent-pose interaction atoms mapped into canonical parent atom "
                "order, then scored by the minimum retained feature count across all enumerated "
                "exact-bond/chiral MCS correspondences. Truncated mapping searches score zero. "
                "The strict target-conditioned track also requires the configured minimum "
                "anchor preservation and strict 3D pharmacophore gate. This affects ranking only "
                "and does not verify an analog pose or binding affinity."
                if anchor_enabled
                else "Not used because no validated interaction-anchor map was supplied."
            ),
            "strict_3d_pharmacophore_semantics": (
                "Up to 50 deterministic ETKDGv3 conformers are generated for parent and "
                "candidate molecules, MMFF94s-optimized when MMFF parameters are available. "
                "Matched ligand feature families are compared after MCS alignment; the strict "
                "3D gate requires both feature-family recall and feature-distance RMSD thresholds. "
                "Unavailable or insufficient matched features fail closed for this strict 3D gate."
            ),
            "pareto_front_semantics": (
                "Pareto fronts are computed over pharmacophore preservation, binding support, "
                "safety triage, routeability, and corpus novelty. A lower front number is "
                "non-dominated within the scored candidate set and is used only as a deterministic "
                "tie-break/selection descriptor, not a probability."
            ),
            "admission_semantics": (
                "strict_2d_pharmacophore requires both configured 2D pharmacophore and "
                "feature-family recall thresholds. Exact same-target activity ligands that "
                "meet the recall threshold may also enter as "
                "same_target_activity_feature_family, but pharmacophore_gate_passed remains "
                "false and no pharmacophore-retention claim is made"
            ),
            "binding_semantics": (
                "direct_retained requires candidate and parent activity with the same "
                "endpoint and source; the most conservative matched-stratum pActivity "
                "delta is used and remains a heterogeneous-assay prioritization signal"
            ),
            "novelty_semantics": "ECFP4 dissimilarity to the parent, not global chemical novelty",
            "structural_similarity_semantics": (
                "Stereo-aware ECFP4 Tanimoto; exact parent identity and less-specific "
                "same-connectivity stereochemical records are excluded"
            ),
            "routeability_semantics": "molecular-complexity proxy, not a retrosynthetic route",
            "safety_triage_semantics": (
                "legacy field name for structural-alert and generic physicochemical triage; "
                "not a candidate-specific skin-safety, sensitization, or toxicology prediction"
            ),
            "pharmacophore_3d_budget_semantics": (
                "The configured candidate budget bounds ligand-based 3D comparisons after all "
                "eligible candidates receive cheap 2D, target, safety, routeability, and Pareto "
                "fields. The deterministic provisional output basket is evaluated first; any "
                "remaining budget follows legacy global priority, Pareto front, evidence tier, "
                "and canonical SMILES. Budget exclusion fails the strict 3D gate only and does "
                "not remove candidates admitted by existing 2D or target tracks."
            ),
        },
        "source_audit": {
            "candidate_library": library_fingerprint,
            "candidate_library_counts": library_counts,
            "activity_evidence": activity_audits,
            "activity_selection": activity_summary,
            "alias_evidence": alias_audits,
            "interaction_anchor_map": (
                _fingerprint(
                    args.interaction_anchor_map,
                    "Interaction-anchor map",
                )
                if args.interaction_anchor_map is not None
                else None
            ),
        },
        "candidate_pool": {
            "merged_unique_molecules": len(seeds),
            "passing_scored_molecules": scoring["passing_count"],
            "scoring_exclusions": scoring["exclusions"],
            "pharmacophore_3d_evaluation": three_d_evaluation,
        },
        "selection_strategy": scoring["selection_strategy"],
        "summary": summary,
        "candidates": rows,
        "artifacts": {
            "json": "substitute_report.json",
            "csv": "substitute_candidates.csv",
            "sdf_3d": "substitute_candidates_3d.sdf",
            "html": "substitute_report.html",
            "markdown": "substitute_report.md",
        },
        "limitations": [
            (
                "The target-conditioned anchor score transfers parent pose-supported atoms "
                "through a conservative 2D graph/feature mapping; ambiguous correspondences use "
                "the minimum preservation score, and the analog pose and affinity are not verified."
                if anchor_enabled
                else "Ligand-feature pharmacophore similarity does not prove preservation of pose-specific protein interactions."
            ),
            "A target-supported feature analogue can be admitted despite failing the strict 2D pharmacophore gate; inspect pharmacophore_gate_passed and admission_bases before interpreting it.",
            "Strict 3D pharmacophore evidence is ligand-based ETKDGv3/MMFF conformer evidence only; unavailable or insufficient matched features fail the strict 3D gate but do not remove hypothesis-only candidates admitted by existing 2D or target-evidence tracks.",
            "Pareto fronts summarize non-dominance across ranking objectives and are not probabilities, binding predictions, or safety claims.",
            "Direct activity values can come from heterogeneous assays; retention tiers require matching endpoint and source, but still do not establish assay equivalence or binding affinity retention.",
            "Candidates without same-target activity evidence are binding hypotheses only.",
            "The balanced candidate basket reserves space for strict 2D pharmacophore hits and CosIng-referenced materials; basket rank is not the same as unstratified global score rank.",
            "CosIng reference status does not establish permitted concentration, formulation compatibility, exposure safety, or regulatory approval.",
            "The safety_triage_score field covers structural alerts and generic physicochemical properties only; it is not a skin-safety or toxicology result.",
            "Routeability is a molecular-complexity proxy unless an external retrosynthesis workflow is run.",
            "Target-specific binding, skin permeation, formulation stability, cytotoxicity, sensitization, and efficacy require prospective experiments.",
            "The 3D candidate budget is a deterministic compute bound, not an evidence score; candidates marked not_evaluated_budget have null 3D metrics and cannot pass the strict 3D or strict target-conditioned gates.",
        ],
    }
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{args.out_dir.name}.staging-",
            dir=args.out_dir.parent,
        )
    )
    committed: list[Path] = []
    try:
        _write_csv(staging / "substitute_candidates.csv", rows)
        _write_sdf(
            staging / "substitute_candidates_3d.sdf",
            report["parent"],
            rows,
            args.seed,
            limit=args.max_3d_candidates,
        )
        _write_html(staging / "substitute_report.html", report)
        _write_markdown(staging / "substitute_report.md", report)
        _write_text_atomic(
            staging / "substitute_report.json",
            json.dumps(report, indent=2, sort_keys=True) + "\n",
        )
        for filename in (
            "substitute_candidates.csv",
            "substitute_candidates_3d.sdf",
            "substitute_report.html",
            "substitute_report.md",
            "substitute_report.json",
        ):
            final = args.out_dir / filename
            os.link(staging / filename, final)
            committed.append(final)
    except BaseException:
        for final in reversed(committed):
            final.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-smiles", required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument(
        "--interaction-anchor-map",
        type=Path,
        help=(
            "Validated Stage 5.5 interaction_anchor_map.json for target-conditioned "
            "pharmacophore ranking"
        ),
    )
    parser.add_argument("--run-id", default="substitute_discovery")
    parser.add_argument(
        "--target-selection-basis",
        choices=("parent_top_prediction", "user_selected_ranked_target"),
        default="parent_top_prediction",
    )
    parser.add_argument("--candidate-library", type=Path, default=Path("data/cosing/cosing.parquet"))
    parser.add_argument("--activity-evidence", action="append", type=Path, default=[])
    parser.add_argument("--alias-evidence", action="append", type=Path, default=[])
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--min-pharmacophore", type=float, default=0.55)
    parser.add_argument("--min-feature-recall", type=float, default=0.60)
    parser.add_argument(
        "--min-3d-feature-recall",
        type=float,
        default=DEFAULT_MIN_3D_FEATURE_RECALL,
        help="Minimum matched 3D pharmacophore feature-family recall for the strict 3D gate",
    )
    parser.add_argument(
        "--max-feature-distance-rmsd",
        type=float,
        default=DEFAULT_MAX_FEATURE_DISTANCE_RMSD,
        help="Maximum matched 3D pharmacophore feature-distance RMSD in Angstrom for the strict 3D gate",
    )
    parser.add_argument(
        "--max-3d-conformers",
        type=int,
        default=DEFAULT_MAX_3D_CONFORMERS,
        help="Maximum deterministic ETKDGv3/MMFF conformers per parent or candidate, capped at 50",
    )
    parser.add_argument(
        "--max-3d-candidates",
        type=int,
        default=DEFAULT_MAX_3D_CANDIDATES,
        help="Maximum candidates receiving deterministic 3D pharmacophore evaluation, capped at 500",
    )
    parser.add_argument(
        "--min-anchor-preservation",
        type=float,
        default=DEFAULT_MIN_ANCHOR_PRESERVATION,
        help="Minimum validated interaction-anchor preservation for the strict target-conditioned track",
    )
    parser.add_argument("--max-structural-similarity", type=float, default=0.98)
    parser.add_argument("--max-potency-loss", type=float, default=1.0)
    parser.add_argument("--max-candidates", type=int, default=50)
    parser.add_argument(
        "--material-track-fraction",
        type=float,
        default=DEFAULT_MATERIAL_TRACK_FRACTION,
        help="Fraction of a multi-track candidate basket reserved for CosIng references",
    )
    parser.add_argument(
        "--pharmacophore-track-fraction",
        type=float,
        default=DEFAULT_PHARMACOPHORE_TRACK_FRACTION,
        help="Fraction of the candidate basket reserved for strict 2D pharmacophore hits",
    )
    parser.add_argument("--max-direct-molecules", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=49242)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    report = run(parse_args(argv))
    print(
        "[substitute-discovery] "
        f"status={report['status']} target={report['target']['target_id']} "
        f"candidates={len(report['candidates'])} claimable=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
