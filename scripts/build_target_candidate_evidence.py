#!/usr/bin/env python3
"""Build the complete target-intent x compound x source-evidence ledger.

This command deliberately retains every matched quantitative source record.  It
does not rank or truncate candidates, and it does not interpret binding as
functional or human-skin evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as ds
from rdkit import Chem, rdBase

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "skinscout.target_candidate_evidence.v1"
_GENE_MAP_ENV = os.environ.get("SKINSCOUT_GENE_MAP_PATH", "").strip()
DEFAULT_GENE_MAP = Path(_GENE_MAP_ENV) if _GENE_MAP_ENV else None


def _clean(value: object) -> object | None:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return value


def _text(value: object) -> str | None:
    value = _clean(value)
    return None if value is None else str(value)


def _integer(value: object) -> int | None:
    value = _clean(value)
    if value is None:
        return None
    return int(value)


def _number(value: object) -> float | None:
    value = _clean(value)
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _boolean(value: object) -> bool | None:
    value = _clean(value)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "yes", "1"}:
        return True
    if normalized in {"false", "no", "0"}:
        return False
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash_payload(*values: object) -> str:
    return hashlib.sha256(_json(list(values)).encode("utf-8")).hexdigest()


def _source_manifest(path: Path) -> tuple[Path | None, dict[str, Any] | None]:
    candidates = [path.parent / "source_manifest.json", path.parent / "manifest.json"]
    for candidate in candidates:
        if candidate.exists():
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"source manifest must be an object: {candidate}")
            return candidate, payload
    return None, None


def _declared_artifact_hash(payload: dict[str, Any] | None, path: Path) -> str | None:
    if payload is None:
        return None
    output = payload.get("output_sha256")
    if isinstance(output, dict):
        value = output.get(path.name)
        if isinstance(value, str):
            return value
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, dict):
        for key in (path.name, path.stem, "activity_evidence"):
            value = artifacts.get(key)
            if isinstance(value, dict) and isinstance(value.get("sha256"), str):
                return value["sha256"]
    return None


def _load_intents(path: Path, gene_map: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("intents") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError("intent JSON must be a list or contain an 'intents' list")
        intents = [dict(row) for row in rows]
    elif path.suffix.lower() in {".csv", ".parquet"}:
        frame = (
            pd.read_csv(path)
            if path.suffix.lower() == ".csv"
            else pd.read_parquet(path)
        )
        intents = [
            {key: _clean(value) for key, value in row.items()}
            for row in frame.to_dict("records")
        ]
    else:
        sys.path.insert(0, str(ROOT / "scripts"))
        from target_intent import build_target_intents  # type: ignore

        intents = [dict(row) for row in build_target_intents(path, gene_map)]
    for row in intents:
        if isinstance(row.get("source_evidence"), str):
            try:
                row["source_evidence"] = json.loads(row["source_evidence"])
            except json.JSONDecodeError:
                pass
        row["source_row_number"] = _integer(row.get("source_row_number"))
        row["target_taxid"] = _integer(row.get("target_taxid"))
        row["docking_eligible"] = _boolean(row.get("docking_eligible"))
    required = {"intent_id", "route", "uniprot_id"}
    for index, row in enumerate(intents):
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"intent row {index} missing fields: {', '.join(missing)}")
    ids = [_text(row.get("intent_id")) for row in intents]
    if any(value is None for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("intent_id values must be nonblank and unique")
    return intents


def _read_matching(
    path: Path, uniprots: Sequence[str], columns: Sequence[str]
) -> pd.DataFrame:
    dataset = ds.dataset(path, format="parquet")
    available = set(dataset.schema.names)
    missing = sorted(set(columns) - available)
    if missing:
        raise ValueError(f"{path} missing columns: {', '.join(missing)}")
    if not uniprots:
        return pd.DataFrame(columns=columns)
    table = dataset.to_table(
        columns=list(columns), filter=ds.field("uniprot").isin(uniprots)
    )
    return table.to_pandas()


def _chunks(values: Iterable[int], size: int = 800) -> Iterator[list[int]]:
    batch: list[int] = []
    for value in sorted(set(values)):
        batch.append(value)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _query_by_ids(
    connection: sqlite3.Connection,
    sql_prefix: str,
    ids: Iterable[int],
) -> list[sqlite3.Row]:
    rows: list[sqlite3.Row] = []
    for batch in _chunks(ids):
        placeholders = ",".join("?" for _ in batch)
        rows.extend(
            connection.execute(f"{sql_prefix} ({placeholders})", batch).fetchall()
        )
    return rows


def _enrich_chembl(
    db_path: Path,
    frame: pd.DataFrame,
    uniprots: Sequence[str],
) -> tuple[
    dict[int, dict[str, Any]],
    dict[int, dict[str, Any]],
    dict[tuple[int, int], list[dict[str, Any]]],
    dict[int, str],
]:
    if not db_path.exists():
        raise FileNotFoundError(f"ChEMBL SQLite DB not found: {db_path}")
    assay_ids = [_integer(v) for v in frame["assay_id"]]
    assay_ids = [v for v in assay_ids if v is not None]
    molregnos = [_integer(v) for v in frame["molecule_molregno"]]
    molregnos = [v for v in molregnos if v is not None]
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=120)
    connection.row_factory = sqlite3.Row
    try:
        target_rows: list[sqlite3.Row] = []
        for batch_start in range(0, len(uniprots), 800):
            batch = list(uniprots[batch_start : batch_start + 800])
            placeholders = ",".join("?" for _ in batch)
            target_rows.extend(
                connection.execute(
                    f"""SELECT DISTINCT td.tid,cs.accession
                         FROM target_dictionary td
                         JOIN target_components tc ON tc.tid=td.tid
                         JOIN component_sequences cs ON cs.component_id=tc.component_id
                         WHERE td.organism='Homo sapiens'
                           AND td.target_type='SINGLE PROTEIN'
                           AND cs.accession IN ({placeholders})""",
                    batch,
                ).fetchall()
            )
        target_uniprot = {int(row["tid"]): str(row["accession"]) for row in target_rows}
        target_ids = set(target_uniprot)
        assay_rows = _query_by_ids(
            connection,
            """SELECT a.assay_id,a.description,a.assay_organism,a.assay_tax_id,
                      a.assay_strain,a.assay_tissue,a.assay_cell_type,
                      a.assay_subcellular_fraction,a.cell_id,a.bao_format,
                      c.cell_name,c.cell_description,c.cell_source_tissue,
                      c.cell_source_organism,c.cell_source_tax_id,c.cellosaurus_id
               FROM assays a LEFT JOIN cell_dictionary c ON c.cell_id=a.cell_id
               WHERE a.assay_id IN""",
            assay_ids,
        )
        mechanisms: dict[tuple[int, int], list[dict[str, Any]]] = {}
        if target_ids:
            placeholders = ",".join("?" for _ in target_ids)
            sql = f"""SELECT dm.mec_id,dm.molregno,dm.tid,dm.mechanism_of_action,
                              dm.action_type,dm.direct_interaction,dm.molecular_mechanism,
                              dm.mechanism_comment,dm.selectivity_comment,
                              dm.binding_site_comment,mr.ref_type,mr.ref_id,mr.ref_url
                       FROM drug_mechanism dm
                       LEFT JOIN mechanism_refs mr ON mr.mec_id=dm.mec_id
                       WHERE dm.tid IN ({placeholders})"""
            grouped: dict[tuple[int, int, int], dict[str, Any]] = {}
            for row in connection.execute(sql, sorted(target_ids)):
                key = (int(row["molregno"]), int(row["tid"]), int(row["mec_id"]))
                item = grouped.setdefault(
                    key,
                    {
                        "mechanism_id": int(row["mec_id"]),
                        "mechanism_of_action": _text(row["mechanism_of_action"]),
                        "action_type": _text(row["action_type"]),
                        "direct_interaction": _boolean(row["direct_interaction"]),
                        "molecular_mechanism": _boolean(row["molecular_mechanism"]),
                        "mechanism_comment": _text(row["mechanism_comment"]),
                        "selectivity_comment": _text(row["selectivity_comment"]),
                        "binding_site_comment": _text(row["binding_site_comment"]),
                        "references": [],
                    },
                )
                if _text(row["ref_type"]):
                    item["references"].append(
                        {
                            "type": _text(row["ref_type"]),
                            "id": _text(row["ref_id"]),
                            "url": _text(row["ref_url"]),
                        }
                    )
            for (molregno, tid, _), item in grouped.items():
                mechanisms.setdefault((molregno, tid), []).append(item)
                molregnos.append(molregno)
        compound_rows = _query_by_ids(
            connection,
            """SELECT md.molregno,md.chembl_id,md.pref_name,
                      cs.canonical_smiles,cs.standard_inchi_key,
                      mh.parent_molregno,pmd.chembl_id AS parent_chembl_id,
                      pcs.canonical_smiles AS parent_smiles,
                      pcs.standard_inchi_key AS parent_inchikey
               FROM molecule_dictionary md
               LEFT JOIN compound_structures cs ON cs.molregno=md.molregno
               LEFT JOIN molecule_hierarchy mh ON mh.molregno=md.molregno
               LEFT JOIN molecule_dictionary pmd ON pmd.molregno=mh.parent_molregno
               LEFT JOIN compound_structures pcs ON pcs.molregno=mh.parent_molregno
               WHERE md.molregno IN""",
            molregnos,
        )
        return (
            {int(row["assay_id"]): dict(row) for row in assay_rows},
            {int(row["molregno"]): dict(row) for row in compound_rows},
            mechanisms,
            target_uniprot,
        )
    finally:
        connection.close()


_CX_EXTENSION_PATTERN = re.compile(r"\|([^|]*)\|")
_INCHIKEY_PATTERN = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
_INCHIKEY_NO_STEREO_BLOCK = "UHFFFAOYSA"

_STEREO_SCOPES = {
    "absolute": "absolute_exact",
    "achiral": "absolute_exact",
    "relative": "relative_stereo",
    "racemic": "racemic_stereo",
    "unspecified": "unspecified_stereo",
}
_STEREO_LIMITATIONS = {
    "relative_stereo": (
        "relative/enhanced stereochemistry; compound_id is stereo-scoped and must "
        "not be used as exact absolute stereoisomer evidence"
    ),
    "racemic_stereo": (
        "racemic enhanced stereochemistry; compound_id is stereo-scoped and must "
        "not be used as exact absolute stereoisomer evidence"
    ),
    "unspecified_stereo": (
        "stereochemistry is unspecified by the source; compound_id is "
        "stereo-scoped and must not be used as exact absolute stereoisomer evidence"
    ),
}


def _cx_extension_text(smiles: str) -> str:
    return ",".join(match.group(1) for match in _CX_EXTENSION_PATTERN.finditer(smiles))


def _cx_tokens(extension: str) -> set[str]:
    return {token.strip().lower() for token in extension.split(",") if token.strip()}


def _cx_relative_marker(extension: str) -> str | None:
    tokens = _cx_tokens(extension)
    if "r" in tokens:
        return "relative"
    if any(token.startswith("&") and ":" in token for token in tokens):
        return "racemic"
    if any(token.startswith("o") and ":" in token for token in tokens):
        return "relative"
    return None


def _inchikey_blocks(key: str | None) -> tuple[str | None, str | None]:
    if not key or not _INCHIKEY_PATTERN.match(key):
        return None, None
    connectivity, stereo, _ = key.split("-")
    return connectivity, stereo


def _stereo_chemistry_class(
    molecule: Chem.Mol,
    extension: str,
    source_key: str | None,
    computed_key: str | None,
) -> str:
    marker = _cx_relative_marker(extension)
    if marker is not None:
        return marker
    group_types = {group.GetGroupType() for group in molecule.GetStereoGroups()}
    if Chem.StereoGroupType.STEREO_AND in group_types:
        return "racemic"
    if Chem.StereoGroupType.STEREO_OR in group_types:
        return "relative"
    if Chem.StereoGroupType.STEREO_ABSOLUTE in group_types:
        return "absolute"
    stereo_infos = Chem.FindPotentialStereo(molecule)
    if any(
        info.specified == Chem.StereoSpecified.Specified for info in stereo_infos
    ):
        _, source_stereo = _inchikey_blocks(source_key)
        _, computed_stereo = _inchikey_blocks(computed_key)
        if (
            source_stereo == _INCHIKEY_NO_STEREO_BLOCK
            and computed_stereo
            and computed_stereo != _INCHIKEY_NO_STEREO_BLOCK
        ):
            # Source key declares "no stereo" while the SMILES carries stereo
            # marks; those marks are unverified and must not become an exact
            # absolute stereoisomer.
            return "unspecified"
        return "absolute"
    if stereo_infos:
        return "unspecified"
    return "achiral"


def _unresolved_structure(
    status: str, smiles: str | None, source_key: str | None
) -> dict[str, Any]:
    return {
        "canonical_isomeric_smiles": None,
        "computed_inchikey": None,
        "computed_connectivity_inchikey": None,
        "stereo_chemistry_class": None,
        "identity_scope": None,
        "structure_identity_limitation": None,
        "compound_id": "unresolved-structure-sha256:"
        + _hash_payload(smiles, source_key),
        "structure_identity_status": status,
    }


def _structure(raw_smiles: object, raw_inchikey: object) -> dict[str, Any]:
    """Canonicalize a source structure without promoting unverified stereo.

    CXSMILES ``|r|``/enhanced-stereo information is preserved in the stored
    canonical string.  Relative, racemic, or unspecified stereochemistry is
    kept connectivity/stereo-scoped (``stereo-scoped-sha256:``) and is never
    emitted as an exact absolute stereoisomer (``structure-sha256:``).
    """
    smiles = _text(raw_smiles)
    source_key = _text(raw_inchikey)
    source_key = source_key.upper() if source_key else None
    if not smiles:
        return _unresolved_structure("missing_structure", smiles, source_key)
    extension = _cx_extension_text(smiles)
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None and extension:
        base = smiles.split("|", 1)[0].strip()
        molecule = Chem.MolFromSmiles(base)
    if molecule is None:
        return _unresolved_structure("invalid_smiles", smiles, source_key)
    groups = molecule.GetStereoGroups()
    if groups or extension:
        canonical = Chem.MolToCXSmiles(molecule)
        if "r" in _cx_tokens(extension) and "|r|" not in canonical:
            # RDKit parses ``|r|`` as plain chirality and drops the relative
            # marker on write; re-attach it so the scope stays explicit.
            canonical = f"{canonical} |r|"
    else:
        canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    try:
        computed_key = Chem.MolToInchiKey(molecule) or None
    except (RuntimeError, ValueError):
        computed_key = None
    source_connectivity, _ = _inchikey_blocks(source_key)
    computed_connectivity, _ = _inchikey_blocks(computed_key)
    if source_key and computed_key:
        if source_connectivity == computed_connectivity:
            status = (
                "validated_match" if source_key == computed_key
                else "source_stereo_mismatch"
            )
        else:
            status = "source_inchikey_mismatch"
    elif computed_key:
        status = "computed_no_source_inchikey"
    else:
        status = "canonical_smiles_only"
    stereo_class = _stereo_chemistry_class(
        molecule, extension, source_key, computed_key
    )
    identity_scope = _STEREO_SCOPES[stereo_class]
    if identity_scope == "absolute_exact":
        compound_id = (
            "structure-sha256:"
            + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        )
    else:
        compound_id = "stereo-scoped-sha256:" + _hash_payload(
            identity_scope, canonical
        )
    return {
        "canonical_isomeric_smiles": canonical,
        "computed_inchikey": computed_key,
        "computed_connectivity_inchikey": computed_connectivity,
        "stereo_chemistry_class": stereo_class,
        "identity_scope": identity_scope,
        "structure_identity_limitation": _STEREO_LIMITATIONS.get(identity_scope),
        "compound_id": compound_id,
        "structure_identity_status": status,
    }


def _taxid_for_organism(organism: object) -> int | None:
    text = (_text(organism) or "").lower()
    return 9606 if text in {"homo sapiens", "human"} else None


def _model_level(assay: dict[str, Any]) -> str:
    context = " ".join(
        _text(assay.get(key)) or ""
        for key in (
            "description",
            "assay_cell_type",
            "cell_name",
            "cell_description",
            "assay_tissue",
        )
    ).lower()
    if "ex vivo" in context or "ex-vivo" in context:
        return "ex_vivo_tissue"
    if any(
        token in context
        for token in ("reconstructed tissue", "skin equivalent", "3d skin")
    ):
        return "reconstructed_tissue"
    if "primary" in context:
        return "primary_cell"
    has_cell = bool(
        _text(assay.get("cell_name")) or _text(assay.get("assay_cell_type"))
    )
    if has_cell and any(
        token in context for token in ("transfect", "recombinant", "reporter")
    ):
        return "engineered_cell"
    if not has_cell and any(
        token in context
        for token in (
            "purified",
            "cell-free",
            "cell free",
            "isolated protein",
            "recombinant protein",
        )
    ):
        return "purified_protein"
    return "unknown"


def _chembl_evidence_axis(source: dict[str, Any], action_type: str | None) -> str:
    endpoint = (_text(source.get("standard_type")) or "").upper()
    assay_type = (_text(source.get("assay_type")) or "").upper()
    if endpoint in {"KD", "KI"} or assay_type == "B":
        return "binding"
    if assay_type == "F" and action_type:
        return "protein_function"
    return "cooccurrence"


def _classify(intent: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from target_intent import classify_evidence  # type: ignore

    result = classify_evidence(intent, evidence)
    if not isinstance(result, dict):
        raise TypeError("classify_evidence must return a dict")
    return {
        "evidence_axis": _text(result.get("evidence_axis")) or "binding",
        "directness": _text(result.get("directness")) or "unknown",
        "direction_relation": _text(result.get("direction_relation")) or "unknown",
        "classification_reason": _text(result.get("reason")),
    }


def _intent_fields(intent: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "intent_id",
        "source_row_id",
        "source_sheet",
        "source_row_number",
        "category",
        "biomarker",
        "full_name",
        "marker_type",
        "route",
        "entity_type",
        "entity_name",
        "gene_symbol",
        "uniprot_id",
        "protein_form",
        "component_of",
        "desired_effect",
        "target_scope",
        "skin_compartment",
        "readout",
        "target_taxid",
        "role",
        "docking_eligible",
    )
    return {key: _clean(intent.get(key)) for key in keys}


def _chembl_records(
    frame: pd.DataFrame,
    intents_by_uniprot: dict[str, list[dict[str, Any]]],
    assays: dict[int, dict[str, Any]],
    compounds: dict[int, dict[str, Any]],
    mechanisms: dict[tuple[int, int], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    structure_cache: dict[tuple[str | None, str | None], dict[str, Any]] = {}
    parent_cache: dict[tuple[str | None, str | None], dict[str, Any]] = {}
    for source in frame.to_dict("records"):
        uniprot = _text(source["uniprot"])
        if uniprot is None:
            continue
        assay_id = _integer(source["assay_id"])
        molregno = _integer(source["molecule_molregno"])
        target_id = _integer(source["target_id"])
        assay = assays.get(assay_id or -1, {})
        compound = compounds.get(molregno or -1, {})
        mechanism_rows = mechanisms.get((molregno or -1, target_id or -1), [])
        raw_smiles, raw_key = (
            _text(source["smiles"]),
            _text(source["standard_inchi_key"]),
        )
        structure = structure_cache.setdefault(
            (raw_smiles, raw_key), _structure(raw_smiles, raw_key)
        )
        parent_smiles = _text(compound.get("parent_smiles"))
        parent_key = _text(compound.get("parent_inchikey"))
        parent = (
            parent_cache.setdefault(
                (parent_smiles, parent_key), _structure(parent_smiles, parent_key)
            )
            if parent_smiles
            else {"compound_id": None}
        )
        relation = _text(source["standard_relation"])
        action_type = _text(source["action_type"])
        mechanism_actions = sorted(
            {
                str(row["action_type"])
                for row in mechanism_rows
                if row.get("action_type")
            }
        )
        pubs = {
            "document_chembl_id": _text(source["document_chembl_id"]),
            "pubmed_id": _text(source["pubmed_id"]),
            "doi": _text(source["doi"]),
            "patent_id": _text(source["patent_id"]),
        }
        for intent in intents_by_uniprot[uniprot]:
            evidence_axis = _chembl_evidence_axis(source, action_type)
            directness = (
                "direct_supported"
                if any(row.get("direct_interaction") is True for row in mechanism_rows)
                else "unknown"
            )
            record: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                **_intent_fields(intent),
                "target_uniprot": uniprot,
                "source_target_id": str(target_id) if target_id is not None else None,
                "source_db": _text(source["source_db"]) or "ChEMBL",
                "source_release": _text(source["source_release"]),
                "source_license": _text(source["source_license"]),
                "source_db_sha256": _text(source["source_db_sha256"]),
                "raw_evidence_id": str(_integer(source["activity_id"])),
                "source_activity_id": _integer(source["activity_id"]),
                "source_assay_id": str(assay_id) if assay_id is not None else None,
                "source_assay_chembl_id": _text(source["assay_chembl_id"]),
                "source_compound_id": _text(source["molecule_chembl_id"]),
                "molecule_chembl_id": _text(source["molecule_chembl_id"]),
                "molecule_pref_name": _text(compound.get("pref_name")),
                "raw_smiles": raw_smiles,
                "raw_inchikey": raw_key,
                **structure,
                "parent_source_compound_id": _text(compound.get("parent_chembl_id")),
                "parent_smiles": parent_smiles,
                "parent_inchikey": parent_key,
                "parent_compound_id": parent.get("compound_id"),
                "endpoint_type": _text(source["standard_type"]),
                "endpoint_value": _number(source["standard_value"]),
                "endpoint_unit": _text(source["standard_units"]),
                "endpoint_relation": relation,
                "endpoint_censored": relation not in (None, "="),
                "pchembl_value": _number(source["pchembl_value"]),
                "activity_action_type": action_type,
                "activity_comment": _text(source["activity_comment"]),
                "data_validity_comment": _text(source["data_validity_comment"]),
                "source_potential_duplicate": _boolean(source["potential_duplicate"]),
                "assay_type": _text(source["assay_type"]),
                "assay_test_type": _text(source["assay_test_type"]),
                "assay_category": _text(source["assay_category"]),
                "assay_confidence_score": _integer(source["assay_confidence_score"]),
                "assay_relationship_type": _text(source["assay_relationship_type"]),
                "assay_source_id": _integer(source["assay_source_id"]),
                "assay_source_name": _text(source["assay_source_name"]),
                "assay_description": _text(assay.get("description")),
                "assay_organism": _text(assay.get("assay_organism")),
                "assay_material_taxid": _integer(assay.get("assay_tax_id")),
                "assay_strain": _text(assay.get("assay_strain")),
                "assay_tissue": _text(assay.get("assay_tissue")),
                "assay_cell_type": _text(assay.get("assay_cell_type")),
                "assay_subcellular_fraction": _text(
                    assay.get("assay_subcellular_fraction")
                ),
                "cell_name": _text(assay.get("cell_name")),
                "cell_description": _text(assay.get("cell_description")),
                "cell_source_tissue": _text(assay.get("cell_source_tissue")),
                "cell_source_organism": _text(assay.get("cell_source_organism")),
                "cell_origin_taxid": _integer(assay.get("cell_source_tax_id")),
                "cellosaurus_id": _text(assay.get("cellosaurus_id")),
                "model_level": _model_level({**source, **assay}),
                "source_target_organism": _text(source["target_organism"]),
                "source_target_taxid": _taxid_for_organism(source["target_organism"]),
                "target_chembl_id": _text(source["target_chembl_id"]),
                "target_pref_name": _text(source["target_pref_name"]),
                "mechanism_action_types_json": _json(mechanism_actions),
                "mechanism_records_json": _json(mechanism_rows),
                "mechanism_direct_interaction": (
                    True
                    if any(
                        row.get("direct_interaction") is True for row in mechanism_rows
                    )
                    else False
                    if mechanism_rows
                    and all(
                        row.get("direct_interaction") is False for row in mechanism_rows
                    )
                    else None
                ),
                "publication_json": _json(pubs),
                "pubchem_cids_json": _json(
                    sorted(
                        {
                            str(ref["id"])
                            for item in mechanism_rows
                            for ref in item.get("references", [])
                            if ref.get("type") == "PubChem" and ref.get("id")
                        }
                    )
                ),
                "document_chembl_id": pubs["document_chembl_id"],
                "publication_year": _integer(source["document_year"]),
                "source_pmid": pubs["pubmed_id"],
                "source_doi": pubs["doi"],
                "source_patent": pubs["patent_id"],
                "source_article_id": pubs["document_chembl_id"],
                "source_input_row_number": None,
                "source_duplicate_group_id": None,
                "source_duplicate_evidence": _boolean(source["potential_duplicate"]),
                "source_origin": _text(source["document_source_name"]),
            }
            classification = _classify(
                intent,
                {
                    **record,
                    "evidence_axis": evidence_axis,
                    "directness": directness,
                    "target_uniprot_id": uniprot,
                    "target_taxid": record["source_target_taxid"],
                    "action_type": action_type,
                    "mechanism_action_types": mechanism_actions,
                    "direct_interaction": record["mechanism_direct_interaction"],
                },
            )
            record.update(classification)
            record["evidence_id"] = "chembl:" + _hash_payload(
                record["source_release"],
                record["source_activity_id"],
                intent["intent_id"],
            )
            records.append(record)
    return records


def _chembl_mechanism_records(
    *,
    intents_by_uniprot: dict[str, list[dict[str, Any]]],
    compounds: dict[int, dict[str, Any]],
    mechanisms: dict[tuple[int, int], list[dict[str, Any]]],
    target_uniprot: dict[int, str],
    source_release: str | None,
    source_license: str | None,
    source_db_sha256: str | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    structure_cache: dict[tuple[str | None, str | None], dict[str, Any]] = {}
    for (molregno, target_id), mechanism_rows in sorted(mechanisms.items()):
        uniprot = target_uniprot.get(target_id)
        if uniprot not in intents_by_uniprot:
            continue
        compound = compounds.get(molregno, {})
        raw_smiles = _text(compound.get("canonical_smiles"))
        raw_key = _text(compound.get("standard_inchi_key"))
        structure = structure_cache.setdefault(
            (raw_smiles, raw_key), _structure(raw_smiles, raw_key)
        )
        if structure["structure_identity_status"] == "missing_structure":
            structure = dict(structure)
            structure["compound_id"] = (
                "unresolved-source-compound-sha256:"
                + _hash_payload("ChEMBL", compound.get("chembl_id"), molregno)
            )
        parent_smiles = _text(compound.get("parent_smiles"))
        parent_key = _text(compound.get("parent_inchikey"))
        parent = (
            structure_cache.setdefault(
                (parent_smiles, parent_key), _structure(parent_smiles, parent_key)
            )
            if parent_smiles
            else {"compound_id": None}
        )
        for mechanism in mechanism_rows:
            references = mechanism.get("references", [])
            pmids = [
                str(ref["id"])
                for ref in references
                if ref.get("type") == "PubMed" and ref.get("id")
            ]
            dois = [
                str(ref["id"])
                for ref in references
                if ref.get("type") == "DOI" and ref.get("id")
            ]
            patents = [
                str(ref["id"])
                for ref in references
                if ref.get("type") in {"Patent", "USPO"} and ref.get("id")
            ]
            pubchem = [
                str(ref["id"])
                for ref in references
                if ref.get("type") == "PubChem" and ref.get("id")
            ]
            action_type = _text(mechanism.get("action_type"))
            directness = (
                "direct_supported"
                if mechanism.get("direct_interaction") is True
                else "unknown"
            )
            for intent in intents_by_uniprot[uniprot]:
                record: dict[str, Any] = {
                    "schema_version": SCHEMA_VERSION,
                    **_intent_fields(intent),
                    "target_uniprot": uniprot,
                    "source_target_id": str(target_id),
                    "source_db": "ChEMBL",
                    "source_release": source_release,
                    "source_license": source_license,
                    "source_db_sha256": source_db_sha256,
                    "raw_evidence_id": f"mechanism:{mechanism['mechanism_id']}",
                    "source_activity_id": None,
                    "source_assay_id": None,
                    "source_assay_chembl_id": None,
                    "source_compound_id": _text(compound.get("chembl_id")),
                    "molecule_chembl_id": _text(compound.get("chembl_id")),
                    "molecule_pref_name": _text(compound.get("pref_name")),
                    "raw_smiles": raw_smiles,
                    "raw_inchikey": raw_key,
                    **structure,
                    "parent_source_compound_id": _text(
                        compound.get("parent_chembl_id")
                    ),
                    "parent_smiles": parent_smiles,
                    "parent_inchikey": parent_key,
                    "parent_compound_id": parent.get("compound_id"),
                    "endpoint_type": None,
                    "endpoint_value": None,
                    "endpoint_unit": None,
                    "endpoint_relation": None,
                    "endpoint_censored": None,
                    "pchembl_value": None,
                    "activity_action_type": None,
                    "activity_comment": None,
                    "data_validity_comment": None,
                    "source_potential_duplicate": False,
                    "assay_type": None,
                    "assay_test_type": None,
                    "assay_category": None,
                    "assay_confidence_score": None,
                    "assay_relationship_type": None,
                    "assay_source_id": None,
                    "assay_source_name": "ChEMBL drug_mechanism",
                    "assay_description": None,
                    "assay_organism": None,
                    "assay_material_taxid": None,
                    "assay_strain": None,
                    "assay_tissue": None,
                    "assay_cell_type": None,
                    "assay_subcellular_fraction": None,
                    "cell_name": None,
                    "cell_description": None,
                    "cell_source_tissue": None,
                    "cell_source_organism": None,
                    "cell_origin_taxid": None,
                    "cellosaurus_id": None,
                    "model_level": "unknown",
                    "source_target_organism": "Homo sapiens",
                    "source_target_taxid": 9606,
                    "target_chembl_id": None,
                    "target_pref_name": None,
                    "mechanism_action_types_json": _json(
                        [action_type] if action_type else []
                    ),
                    "mechanism_records_json": _json([mechanism]),
                    "mechanism_direct_interaction": _boolean(
                        mechanism.get("direct_interaction")
                    ),
                    "publication_json": _json(
                        {"pubmed_ids": pmids, "dois": dois, "patents": patents}
                    ),
                    "pubchem_cids_json": _json(sorted(set(pubchem))),
                    "document_chembl_id": None,
                    "publication_year": None,
                    "source_pmid": pmids[0] if pmids else None,
                    "source_doi": dois[0] if dois else None,
                    "source_patent": patents[0] if patents else None,
                    "source_article_id": None,
                    "source_input_row_number": None,
                    "source_duplicate_group_id": None,
                    "source_duplicate_evidence": False,
                    "source_origin": "ChEMBL drug_mechanism",
                }
                record.update(
                    _classify(
                        intent,
                        {
                            **record,
                            "evidence_axis": "protein_function",
                            "directness": directness,
                            "target_uniprot_id": uniprot,
                            "target_taxid": 9606,
                            "action_type": action_type,
                        },
                    )
                )
                record["evidence_id"] = "chembl-mechanism:" + _hash_payload(
                    source_release, mechanism["mechanism_id"], intent["intent_id"]
                )
                records.append(record)
    return records


def _bindingdb_records(
    frame: pd.DataFrame,
    intents_by_uniprot: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    structure_cache: dict[tuple[str | None, str | None], dict[str, Any]] = {}
    for source in frame.to_dict("records"):
        uniprot = _text(source["uniprot"])
        if uniprot is None:
            continue
        raw_smiles, raw_key = (
            _text(source["ligand_smiles"]),
            _text(source["ligand_inchikey"]),
        )
        structure = structure_cache.setdefault(
            (raw_smiles, raw_key), _structure(raw_smiles, raw_key)
        )
        relation = _text(source["relation"])
        pubs = {
            "pubmed_id": _text(source["source_pmid"]),
            "doi": _text(source["source_doi"]),
            "patent_id": _text(source["source_patent"]),
            "article_id": _text(source["source_article_id"]),
        }
        for intent in intents_by_uniprot[uniprot]:
            record: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                **_intent_fields(intent),
                "target_uniprot": uniprot,
                "source_target_id": uniprot,
                "source_db": "BindingDB",
                "source_release": _text(source["source_release"]),
                "source_license": _text(source["source_license"]),
                "source_db_sha256": None,
                "raw_evidence_id": _text(source["evidence_id"]),
                "source_activity_id": None,
                "source_assay_id": _text(source["source_article_id"]),
                "source_assay_chembl_id": None,
                "source_compound_id": _text(source["ligand_id"]),
                "molecule_chembl_id": None,
                "molecule_pref_name": None,
                "raw_smiles": raw_smiles,
                "raw_inchikey": raw_key,
                **structure,
                "parent_source_compound_id": None,
                "parent_smiles": None,
                "parent_inchikey": None,
                "parent_compound_id": None,
                "endpoint_type": _text(source["affinity_type"]),
                "endpoint_value": _number(source["affinity_value"]),
                "endpoint_unit": _text(source["affinity_unit"]),
                "endpoint_relation": relation,
                "endpoint_censored": _boolean(source["censor"]),
                "pchembl_value": None,
                "activity_action_type": None,
                "activity_comment": None,
                "data_validity_comment": None,
                "source_potential_duplicate": _boolean(source["duplicate_evidence"]),
                "assay_type": "binding",
                "assay_test_type": None,
                "assay_category": None,
                "assay_confidence_score": None,
                "assay_relationship_type": None,
                "assay_source_id": None,
                "assay_source_name": None,
                "assay_description": None,
                "assay_organism": _text(source["organism"]),
                "assay_material_taxid": _taxid_for_organism(source["organism"]),
                "assay_strain": None,
                "assay_tissue": None,
                "assay_cell_type": None,
                "assay_subcellular_fraction": None,
                "cell_name": None,
                "cell_description": None,
                "cell_source_tissue": None,
                "cell_source_organism": None,
                "cell_origin_taxid": None,
                "cellosaurus_id": None,
                "model_level": "unknown",
                "source_target_organism": _text(source["organism"]),
                "source_target_taxid": _taxid_for_organism(source["organism"]),
                "target_chembl_id": None,
                "target_pref_name": None,
                "mechanism_action_types_json": "[]",
                "mechanism_records_json": "[]",
                "mechanism_direct_interaction": None,
                "publication_json": _json(pubs),
                "pubchem_cids_json": "[]",
                "document_chembl_id": None,
                "publication_year": None,
                "source_pmid": pubs["pubmed_id"],
                "source_doi": pubs["doi"],
                "source_patent": pubs["patent_id"],
                "source_article_id": pubs["article_id"],
                "source_input_row_number": _integer(source["input_row_number"]),
                "source_duplicate_group_id": _text(source["duplicate_group_id"]),
                "source_duplicate_evidence": _boolean(source["duplicate_evidence"]),
                "source_origin": _text(source["source_origin"]),
            }
            classification = _classify(intent, {**record, "action_type": None})
            # BindingDB has no functional action direction.  Fail closed even if a
            # future classifier mistakenly infers one from the desired effect.
            record.update(classification)
            record["evidence_axis"] = "binding"
            record["direction_relation"] = "unknown"
            if record["directness"] not in {"direct_supported", "unknown"}:
                record["directness"] = "unknown"
            record["classification_reason"] = (
                "BindingDB quantitative binding record; functional direction and "
                "human skin/cell activity are not established by this record"
            )
            record["evidence_id"] = "bindingdb:" + _hash_payload(
                record["source_release"], record["raw_evidence_id"], intent["intent_id"]
            )
            records.append(record)
    return records


def _mark_cross_source_duplicates(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        frame["cross_source_duplicate_group_id"] = pd.Series(dtype="object")
        frame["possible_cross_source_duplicate"] = pd.Series(dtype="bool")
        return frame
    group_ids = []
    for row in frame.to_dict("records"):
        publication = _text(row.get("source_doi")) or _text(row.get("source_pmid"))
        if publication is None:
            publication = _text(row.get("source_patent")) or _text(
                row.get("source_article_id")
            )
        group_ids.append(
            _hash_payload(
                row.get("intent_id"),
                row.get("compound_id"),
                row.get("endpoint_type"),
                row.get("endpoint_relation"),
                row.get("endpoint_value"),
                row.get("endpoint_unit"),
                publication,
            )
        )
    frame = frame.copy()
    frame["cross_source_duplicate_group_id"] = group_ids
    source_counts = frame.groupby("cross_source_duplicate_group_id")[
        "source_db"
    ].transform("nunique")
    frame["possible_cross_source_duplicate"] = source_counts.gt(1)
    return frame


def _unique_json(values: Iterable[object]) -> str:
    cleaned = sorted({_text(value) for value in values if _text(value) is not None})
    return _json(cleaned)


def _summary(ledger: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "intent_id",
        "source_row_id",
        "category",
        "biomarker",
        "entity_name",
        "gene_symbol",
        "target_uniprot",
        "desired_effect",
        "route",
        "compound_id",
        "canonical_isomeric_smiles",
        "computed_inchikey",
        "stereo_chemistry_class",
        "identity_scope",
        "structure_identity_status",
        "structure_identity_statuses_json",
        "identity_scopes_json",
        "structure_identity_limitations_json",
        "source_identity_mismatch_count",
        "source_connectivity_mismatch_count",
        "source_stereo_mismatch_count",
        "raw_inchikeys_json",
        "computed_inchikeys_json",
        "parent_compound_id",
        "parent_source_compound_id",
        "parent_compound_ids_json",
        "parent_source_compound_ids_json",
        "molecule_pref_name",
        "molecule_chembl_ids_json",
        "source_compound_ids_json",
        "source_dbs_json",
        "source_releases_json",
        "source_licenses_json",
        "source_evidence_hashes_json",
        "source_target_ids_json",
        "evidence_count",
        "unique_source_record_count",
        "unique_assay_count",
        "unique_publication_count",
        "endpoint_types_json",
        "endpoint_relations_json",
        "action_types_json",
        "activity_action_types_json",
        "mechanism_action_types_json",
        "evidence_axes_json",
        "directness_json",
        "direction_relations_json",
        "pmids_json",
        "dois_json",
        "patents_json",
        "pubchem_cids_json",
        "assay_ids_json",
        "assay_source_ids_json",
        "assay_source_names_json",
        "activity_ids_json",
        "possible_cross_source_duplicate_count",
        "assay_organisms_json",
        "assay_material_taxids_json",
        "cell_origin_taxids_json",
        "model_levels_json",
        "assay_tissues_json",
        "assay_confidence_scores_json",
        "data_validity_comments_json",
        "censored_evidence_count",
    ]
    if ledger.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    keys = ["intent_id", "compound_id"]
    for (_, _), group in ledger.groupby(keys, dropna=False, sort=True):
        first = group.iloc[0]
        identity_statuses = {
            str(value) for value in group["structure_identity_status"] if _text(value)
        }
        identity_severity = {
            "missing_structure": 6,
            "invalid_smiles": 5,
            "source_inchikey_mismatch": 4,
            "source_stereo_mismatch": 3,
            "canonical_smiles_only": 2,
            "computed_no_source_inchikey": 1,
            "validated_match": 0,
        }
        identity_status = max(
            identity_statuses,
            key=lambda value: (identity_severity.get(value, 7), value),
        )
        publication_ids = []
        for row in group.to_dict("records"):
            publication_ids.append(
                _text(row.get("source_doi"))
                or _text(row.get("source_pmid"))
                or _text(row.get("source_patent"))
                or _text(row.get("source_article_id"))
            )
        activity_actions: list[str] = []
        mechanism_actions: list[str] = []
        pubchem_cids: list[str] = []
        for row in group.to_dict("records"):
            if _text(row.get("activity_action_type")):
                activity_actions.append(str(row["activity_action_type"]))
            try:
                mechanism_actions.extend(
                    json.loads(str(row["mechanism_action_types_json"]))
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
            try:
                pubchem_cids.extend(json.loads(str(row["pubchem_cids_json"])))
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        rows.append(
            {
                "intent_id": first["intent_id"],
                "source_row_id": first["source_row_id"],
                "category": first["category"],
                "biomarker": first["biomarker"],
                "entity_name": first["entity_name"],
                "gene_symbol": first["gene_symbol"],
                "target_uniprot": first["target_uniprot"],
                "desired_effect": first["desired_effect"],
                "route": first["route"],
                "compound_id": first["compound_id"],
                "canonical_isomeric_smiles": first["canonical_isomeric_smiles"],
                "computed_inchikey": first["computed_inchikey"],
                "stereo_chemistry_class": first["stereo_chemistry_class"],
                "identity_scope": first["identity_scope"],
                "structure_identity_status": identity_status,
                "structure_identity_statuses_json": _json(sorted(identity_statuses)),
                "identity_scopes_json": _unique_json(group["identity_scope"]),
                "structure_identity_limitations_json": _unique_json(
                    group["structure_identity_limitation"]
                ),
                "source_connectivity_mismatch_count": int(
                    (
                        group["structure_identity_status"] == "source_inchikey_mismatch"
                    ).sum()
                ),
                "source_stereo_mismatch_count": int(
                    (
                        group["structure_identity_status"] == "source_stereo_mismatch"
                    ).sum()
                ),
                "source_identity_mismatch_count": int(
                    group["structure_identity_status"].isin(
                        {"source_inchikey_mismatch", "source_stereo_mismatch"}
                    ).sum()
                ),
                "raw_inchikeys_json": _unique_json(group["raw_inchikey"]),
                "computed_inchikeys_json": _unique_json(group["computed_inchikey"]),
                "parent_compound_id": next(
                    (
                        _text(value)
                        for value in group["parent_compound_id"]
                        if _text(value)
                    ),
                    None,
                ),
                "parent_source_compound_id": next(
                    (
                        _text(value)
                        for value in group["parent_source_compound_id"]
                        if _text(value)
                    ),
                    None,
                ),
                "parent_compound_ids_json": _unique_json(group["parent_compound_id"]),
                "parent_source_compound_ids_json": _unique_json(
                    group["parent_source_compound_id"]
                ),
                "molecule_pref_name": next(
                    (_text(v) for v in group["molecule_pref_name"] if _text(v)), None
                ),
                "molecule_chembl_ids_json": _unique_json(group["molecule_chembl_id"]),
                "source_compound_ids_json": _unique_json(group["source_compound_id"]),
                "source_dbs_json": _unique_json(group["source_db"]),
                "source_releases_json": _unique_json(group["source_release"]),
                "source_licenses_json": _unique_json(group["source_license"]),
                "source_evidence_hashes_json": _unique_json(
                    group["source_evidence_file_sha256"]
                ),
                "source_target_ids_json": _unique_json(group["source_target_id"]),
                "evidence_count": len(group),
                "unique_source_record_count": group["raw_evidence_id"].nunique(
                    dropna=True
                ),
                "unique_assay_count": group["source_assay_id"].nunique(dropna=True),
                "unique_publication_count": len({v for v in publication_ids if v}),
                "endpoint_types_json": _unique_json(group["endpoint_type"]),
                "endpoint_relations_json": _unique_json(group["endpoint_relation"]),
                "action_types_json": _unique_json(
                    [*activity_actions, *mechanism_actions]
                ),
                "activity_action_types_json": _unique_json(activity_actions),
                "mechanism_action_types_json": _unique_json(mechanism_actions),
                "evidence_axes_json": _unique_json(group["evidence_axis"]),
                "directness_json": _unique_json(group["directness"]),
                "direction_relations_json": _unique_json(group["direction_relation"]),
                "pmids_json": _unique_json(group["source_pmid"]),
                "dois_json": _unique_json(group["source_doi"]),
                "patents_json": _unique_json(group["source_patent"]),
                "pubchem_cids_json": _unique_json(pubchem_cids),
                "assay_ids_json": _unique_json(group["source_assay_id"]),
                "assay_source_ids_json": _unique_json(group["assay_source_id"]),
                "assay_source_names_json": _unique_json(group["assay_source_name"]),
                "activity_ids_json": _unique_json(group["source_activity_id"]),
                "possible_cross_source_duplicate_count": int(
                    group["possible_cross_source_duplicate"].sum()
                ),
                "assay_organisms_json": _unique_json(group["assay_organism"]),
                "assay_material_taxids_json": _unique_json(
                    group["assay_material_taxid"]
                ),
                "cell_origin_taxids_json": _unique_json(group["cell_origin_taxid"]),
                "model_levels_json": _unique_json(group["model_level"]),
                "assay_tissues_json": _unique_json(group["assay_tissue"]),
                "assay_confidence_scores_json": _unique_json(
                    group["assay_confidence_score"]
                ),
                "data_validity_comments_json": _unique_json(
                    group["data_validity_comment"]
                ),
                "censored_evidence_count": int(
                    group["endpoint_censored"].fillna(False).sum()
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _coverage(intents: list[dict[str, Any]], ledger: pd.DataFrame) -> pd.DataFrame:
    counts: dict[str, dict[str, int]] = {}
    if not ledger.empty:
        for intent_id, group in ledger.groupby("intent_id", sort=False):
            publications = {
                _text(row.get("source_doi"))
                or _text(row.get("source_pmid"))
                or _text(row.get("source_patent"))
                or _text(row.get("source_article_id"))
                for row in group.to_dict("records")
            }
            publications.discard(None)
            counts[str(intent_id)] = {
                "evidence_count": len(group),
                "candidate_count": int(group["compound_id"].nunique(dropna=True)),
                "source_count": int(group["source_db"].nunique(dropna=True)),
                "publication_count": len(publications),
            }
    rows = []
    for intent in intents:
        intent_id = str(intent["intent_id"])
        stats = counts.get(intent_id, {})
        rows.append(
            {
                **_intent_fields(intent),
                "evidence_count": int(stats.get("evidence_count", 0)),
                "candidate_count": int(stats.get("candidate_count", 0)),
                "source_count": int(stats.get("source_count", 0)),
                "publication_count": int(stats.get("publication_count", 0)),
                "evidence_status": "evidence_found"
                if stats
                else "no_local_source_evidence",
                "coverage_reason": (
                    "nonprotein endpoint; retained for endpoint evidence workflow"
                    if intent.get("route") == "endpoint"
                    else "no quantitative records matched either local source"
                    if not stats
                    else "quantitative source records retained without ranking"
                ),
            }
        )
    return pd.DataFrame(rows)


def _write_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        if path.suffix == ".parquet":
            frame.to_parquet(temporary, index=False)
        else:
            frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.gene_map is None:
        raise SystemExit(
            "--gene-map 또는 SKINSCOUT_GENE_MAP_PATH 환경변수가 필요합니다"
        )
    intents = _load_intents(args.intents, args.gene_map)
    intents_by_uniprot: dict[str, list[dict[str, Any]]] = {}
    for intent in intents:
        uniprot = _text(intent.get("uniprot_id"))
        if uniprot:
            intents_by_uniprot.setdefault(uniprot, []).append(intent)
    uniprots = sorted(intents_by_uniprot)
    chembl_columns = [
        "target_chembl_id",
        "uniprot",
        "molecule_chembl_id",
        "smiles",
        "source_db",
        "source_release",
        "source_license",
        "source_db_sha256",
        "activity_id",
        "assay_id",
        "assay_chembl_id",
        "assay_type",
        "assay_test_type",
        "assay_category",
        "assay_confidence_score",
        "assay_relationship_type",
        "assay_source_id",
        "assay_source_name",
        "target_id",
        "target_organism",
        "target_pref_name",
        "molecule_molregno",
        "standard_inchi_key",
        "document_chembl_id",
        "document_year",
        "pubmed_id",
        "doi",
        "patent_id",
        "document_source_name",
        "standard_relation",
        "standard_type",
        "standard_value",
        "standard_units",
        "pchembl_value",
        "data_validity_comment",
        "potential_duplicate",
        "activity_comment",
        "action_type",
    ]
    binding_columns = [
        "evidence_id",
        "duplicate_group_id",
        "duplicate_evidence",
        "input_row_number",
        "source_origin",
        "source_release",
        "source_license",
        "ligand_smiles",
        "ligand_inchikey",
        "ligand_id",
        "uniprot",
        "organism",
        "affinity_type",
        "relation",
        "censor",
        "affinity_value",
        "affinity_unit",
        "source_pmid",
        "source_doi",
        "source_patent",
        "source_article_id",
    ]
    chembl_manifest_path, chembl_manifest = _source_manifest(args.chembl_evidence)
    binding_manifest_path, binding_manifest = _source_manifest(args.bindingdb_evidence)
    chembl_hash = _sha256(args.chembl_evidence)
    binding_hash = _sha256(args.bindingdb_evidence)
    for path, observed, payload in (
        (args.chembl_evidence, chembl_hash, chembl_manifest),
        (args.bindingdb_evidence, binding_hash, binding_manifest),
    ):
        expected = _declared_artifact_hash(payload, path)
        if expected and expected != observed:
            raise RuntimeError(
                f"source manifest hash mismatch for {path}: {expected} != {observed}"
            )
    chembl = _read_matching(args.chembl_evidence, uniprots, chembl_columns)
    binding = _read_matching(args.bindingdb_evidence, uniprots, binding_columns)
    assays, compounds, mechanisms, target_uniprot = _enrich_chembl(
        args.chembl_sqlite, chembl, uniprots
    )
    records = _chembl_records(chembl, intents_by_uniprot, assays, compounds, mechanisms)
    records.extend(
        _chembl_mechanism_records(
            intents_by_uniprot=intents_by_uniprot,
            compounds=compounds,
            mechanisms=mechanisms,
            target_uniprot=target_uniprot,
            source_release=next(
                (_text(v) for v in chembl["source_release"] if _text(v)), None
            ),
            source_license=next(
                (_text(v) for v in chembl["source_license"] if _text(v)), None
            ),
            source_db_sha256=next(
                (_text(v) for v in chembl["source_db_sha256"] if _text(v)), None
            ),
        )
    )
    records.extend(_bindingdb_records(binding, intents_by_uniprot))
    ledger = _mark_cross_source_duplicates(pd.DataFrame(records))
    del records
    if ledger.empty:
        raise RuntimeError("no matching candidate evidence records were found")
    ledger["source_evidence_file_sha256"] = ledger["source_db"].map(
        {"ChEMBL": chembl_hash, "BindingDB": binding_hash}
    )
    ledger.sort_values(
        ["intent_id", "compound_id", "source_db", "evidence_id"], inplace=True
    )
    summary = _summary(ledger)
    coverage = _coverage(intents, ledger)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.unlink(missing_ok=True)
    manifest_path.with_suffix(".json.tmp").unlink(missing_ok=True)
    outputs = {
        "candidate_evidence.parquet": ledger,
        "candidate_source_summary.parquet": summary,
        "candidate_source_summary.csv": summary,
        "evidence_coverage.csv": coverage,
    }
    for name, frame in outputs.items():
        _write_atomic(frame, args.out_dir / name)

    source_counts = Counter(str(value) for value in ledger["source_db"])
    target_intent_path = ROOT / "scripts" / "target_intent.py"
    registry_path: Path | None = None
    for intent in intents:
        source_evidence = intent.get("source_evidence")
        if isinstance(source_evidence, dict) and _text(
            source_evidence.get("registry_path")
        ):
            candidate = Path(str(source_evidence["registry_path"]))
            registry_path = candidate if candidate.is_absolute() else ROOT / candidate
            break
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "policy": {
            "candidate_scope": "all locally matched records; no top-N truncation",
            "identity": (
                "compound_id is SHA-256 of RDKit canonical isomeric SMILES when "
                "stereochemistry is absolute and source-verified "
                "(structure-sha256:); relative/racemic/unspecified stereochemistry "
                "stays connectivity/stereo-scoped (stereo-scoped-sha256:) with an "
                "explicit limitation and is never exact absolute evidence; raw and "
                "parent source identities retained separately"
            ),
            "bindingdb_interpretation": "binding evidence only; functional direction and human skin/cell activity remain unknown",
            "duplicate_policy": "records retained; source duplicates and possible cross-source duplicate groups labeled, never silently collapsed",
            "missing_policy": "unknown fields remain null and are not imputed",
        },
        "inputs": {
            "intents": {
                "path": str(args.intents),
                "sha256": _sha256(args.intents),
                "rows": len(intents),
            },
            "gene_map": {"path": str(args.gene_map), "sha256": _sha256(args.gene_map)},
            "intent_registry": {
                "path": str(registry_path) if registry_path else None,
                "sha256": _sha256(registry_path)
                if registry_path and registry_path.exists()
                else None,
            },
            "chembl_evidence": {
                "path": str(args.chembl_evidence),
                "sha256": chembl_hash,
                "manifest": str(chembl_manifest_path) if chembl_manifest_path else None,
                "manifest_sha256": _sha256(chembl_manifest_path)
                if chembl_manifest_path
                else None,
                "matched_source_rows": len(chembl),
            },
            "bindingdb_evidence": {
                "path": str(args.bindingdb_evidence),
                "sha256": binding_hash,
                "manifest": str(binding_manifest_path)
                if binding_manifest_path
                else None,
                "manifest_sha256": _sha256(binding_manifest_path)
                if binding_manifest_path
                else None,
                "matched_source_rows": len(binding),
            },
            "chembl_sqlite": {
                "path": str(args.chembl_sqlite),
                "bytes": args.chembl_sqlite.stat().st_size,
                "sha256": next(
                    (str(v) for v in ledger["source_db_sha256"] if _text(v)), None
                ),
                "identity_source": "ChEMBL source manifest and per-evidence source_db_sha256",
            },
        },
        "software": {
            "builder": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "target_intent": {
                "path": str(target_intent_path),
                "sha256": _sha256(target_intent_path),
            },
            "python": sys.version,
            "python_executable": sys.executable,
            "pandas": pd.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "counts": {
            "input_intents": len(intents),
            "protein_intents": sum(
                _text(row.get("uniprot_id")) is not None for row in intents
            ),
            "intents_with_evidence": int((coverage["evidence_count"] > 0).sum()),
            "intents_without_evidence": int((coverage["evidence_count"] == 0).sum()),
            "ledger_rows": len(ledger),
            "candidate_source_summary_rows": len(summary),
            "unique_candidate_ids": int(ledger["compound_id"].nunique(dropna=True)),
            "unique_exact_structures": int(
                ledger.loc[
                    ledger["compound_id"].str.startswith("structure-sha256:", na=False),
                    "compound_id",
                ].nunique()
            ),
            "unique_stereo_scoped_candidate_ids": int(
                ledger.loc[
                    ledger["compound_id"].str.startswith(
                        "stereo-scoped-sha256:", na=False
                    ),
                    "compound_id",
                ].nunique()
            ),
            "unresolved_structure_candidate_ids": int(
                ledger.loc[
                    ~ledger["compound_id"].str.startswith(
                        "structure-sha256:", na=False
                    ),
                    "compound_id",
                ].nunique()
            ),
            "source_rows": dict(sorted(source_counts.items())),
            "chembl_activity_rows": int(ledger["source_activity_id"].notna().sum()),
            "chembl_mechanism_rows": int(
                ledger["raw_evidence_id"].astype(str).str.startswith("mechanism:").sum()
            ),
            "bindingdb_rows": int((ledger["source_db"] == "BindingDB").sum()),
            "structure_identity_rows": {
                str(key): int(value)
                for key, value in ledger["structure_identity_status"]
                .value_counts(dropna=False)
                .items()
            },
            "connectivity_mismatch_rows": int(
                (ledger["structure_identity_status"] == "source_inchikey_mismatch").sum()
            ),
            "stereo_mismatch_rows": int(
                (ledger["structure_identity_status"] == "source_stereo_mismatch").sum()
            ),
            "identity_scope_rows": {
                str(key): int(value)
                for key, value in ledger["identity_scope"]
                .value_counts(dropna=False)
                .items()
            },
            "direction_relation_rows": {
                str(key): int(value)
                for key, value in ledger["direction_relation"]
                .value_counts(dropna=False)
                .items()
            },
            "possible_cross_source_duplicate_rows": int(
                ledger["possible_cross_source_duplicate"].sum()
            ),
        },
        "artifacts": {},
    }
    for name, output_frame in outputs.items():
        output_path = args.out_dir / name
        manifest["artifacts"][name] = {
            "path": name,
            "bytes": output_path.stat().st_size,
            "sha256": _sha256(output_path),
            "rows": len(output_frame),
        }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, manifest_path)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--intents",
        type=Path,
        required=True,
        help="Biomarker XLSX or intent registry CSV/JSON/Parquet",
    )
    parser.add_argument("--gene-map", type=Path, default=DEFAULT_GENE_MAP)
    parser.add_argument("--chembl-evidence", type=Path, required=True)
    parser.add_argument("--bindingdb-evidence", type=Path, required=True)
    parser.add_argument("--chembl-sqlite", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    manifest = build(parse_args(argv))
    counts = manifest["counts"]
    print(
        f"wrote {counts['ledger_rows']:,} evidence rows for "
        f"{counts['intents_with_evidence']}/{counts['input_intents']} intents and "
        f"{counts['unique_exact_structures']:,} exact structures"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
