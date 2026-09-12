#!/usr/bin/env python3
"""Ligand-similarity target retrieval with leakage-controlled evidence modes.

For each ChEMBL human-activity ligand:
  similarity(t)  = tanimoto(query_ECFP4, ligand_ECFP4)
Aggregate per UniProt target via max similarity across its known actives, then
emit a target ranking. ``retrieval`` mode preserves the historical baseline.
``leave-query-out`` and ``temporal`` modes remove exact/high-similarity reference
ligands so known-target evaluation cannot be satisfied by direct answer lookup.

This remains a 2D ECFP4 approximation of the Daina & Zoete baseline. The
quality-aware score is an opt-in ablation, not a calibrated probability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs
from rdkit.Chem import inchi, rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("stage3.daina")
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
# Stage 1 protonates the ligand at pH 7.2-7.6 for docking, and that ionised
# species reaches Stage 3. The reference sets store neutral structures, so the
# query is neutralised back with the same canonicalisation the index uses
# (`build_activity_retrieval_index.standardize_parent`).
EVIDENCE_MODES = ("retrieval", "leave-query-out", "temporal")
SCORING_METHODS = ("max-similarity", "quality-hybrid")
QUALITY_POLICIES = ("legacy", "high-confidence", "claim-grade")
HIGH_CONFIDENCE_COLUMNS = {
    "pchembl",
    "standard_relation",
    "assay_confidence_score",
    "potential_duplicate",
    "data_validity_comment",
}
CLAIM_GRADE_COLUMNS = HIGH_CONFIDENCE_COLUMNS | {
    "assay_type",
    "target_type",
    "assay_relationship_type",
    "standard_type",
    "standard_value",
    "standard_units",
    "document_type",
    "document_year",
    "assay_source_name",
    "document_source_name",
    "assay_variant_id",
}
BIT_REVERSE = np.array(
    [int(f"{value:08b}"[::-1], 2) for value in range(256)],
    dtype=np.uint8,
)


def _remove_outputs(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def _write_tsv_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, sep="\t", index=False)
    tmp.replace(path)


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def query_features(
    sdf: Path,
) -> tuple["DataStructs.ExplicitBitVect", str]:
    sup = Chem.SDMolSupplier(str(sdf), removeHs=False)
    for m in sup:
        if m is not None:
            try:
                from build_activity_retrieval_index import standardize_parent

                parent = standardize_parent(Chem.RemoveHs(m))
                key = inchi.MolToInchiKey(parent).split("-", 1)[0]
            except Exception as exc:
                raise SystemExit(
                    f"Unable to derive standardized query InChIKey from {sdf}"
                ) from exc
            if len(key) != 14 or not key.isalpha():
                raise SystemExit(
                    f"Unable to derive valid query InChIKey connectivity block from {sdf}"
                )
            # The reference fingerprints - the ChEMBL mirror and the retrieval
            # index alike - are built from neutral SMILES, so hydrogens are
            # implicit and nothing carries a formal charge. The run's SDF is a
            # 3D conformer of the pH-7.4 ion, and Morgan encodes both explicit
            # hydrogens and formal charge as atom invariants.
            #
            # Measured on the 53 runs on disk before this was fixed: 24 of them
            # had a query that did not match its own compound as the reference
            # set stores it. Hydroquinone (charge -2) scored 0.231 against
            # itself, kojic acid 0.643, alpha-arbutin 0.750. Minoxidil was worse
            # than a similarity loss - protonation changed its InChIKey
            # connectivity block, so the leave-query-out identity filter matched
            # none of its own ChEMBL rows and left them in the reference set.
            canonical = Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True)
            fingerprint = MORGAN_GENERATOR.GetFingerprint(parent)
            return fingerprint, key
    raise SystemExit(f"No mol in {sdf}")


def query_fp(sdf: Path) -> "DataStructs.ExplicitBitVect":
    return query_features(sdf)[0]


def _parse_uint64_word(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("boolean is not a uint64 fingerprint word")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid uint64 fingerprint word: {value!r}") from exc
    if parsed < 0 or parsed > np.iinfo(np.uint64).max:
        raise ValueError(f"uint64 fingerprint word out of range: {value!r}")
    return parsed


def fp_from_uint64_row(row: list[object]) -> "DataStructs.ExplicitBitVect":
    if len(row) != 32:
        raise ValueError(f"expected 32 uint64 words for 2048-bit ECFP4, got {len(row)}")
    words = np.array(
        [_parse_uint64_word(value) for value in row],
        dtype="<u8",
    )
    # Stored bytes use np.packbits' MSB-first convention, while RDKit's binary
    # text loader expects bit 0 in the least-significant position of each byte.
    # Reversing each byte keeps conversion in C/NumPy instead of setting 2,048
    # bits in Python for every reference molecule.
    rdkit_bytes = BIT_REVERSE[words.view(np.uint8)].tobytes()
    return DataStructs.CreateFromBinaryText(rdkit_bytes)


def validate_nonblank_text_column(
    df: pd.DataFrame, path: Path, label: str, column: str
) -> None:
    invalid_indexes: list[int] = []
    for idx, value in df[column].items():
        if pd.isna(value) or str(value).strip() == "":
            invalid_indexes.append(int(idx))
    if invalid_indexes:
        shown = ",".join(str(idx) for idx in invalid_indexes[:10])
        suffix = "..." if len(invalid_indexes) > 10 else ""
        raise SystemExit(
            f"{label} column '{column}' contains blank values at "
            f"row index(es) {shown}{suffix}: {path}"
        )


def load_fingerprints(path: Path) -> pd.DataFrame:
    try:
        fp_df = pd.read_parquet(path, columns=["molecule_chembl_id", "bitvec"])
    except Exception as exc:
        raise SystemExit(
            "ChEMBL fingerprint parquet missing required columns "
            f"['molecule_chembl_id', 'bitvec']: {path}"
        ) from exc
    if fp_df.empty:
        raise SystemExit(f"ChEMBL fingerprint parquet contains no ligands: {path}")
    validate_nonblank_text_column(
        fp_df, path, "ChEMBL fingerprint parquet", "molecule_chembl_id"
    )
    normalized_ids = fp_df["molecule_chembl_id"].astype(str).str.strip()
    duplicate_ids = sorted(set(normalized_ids[normalized_ids.duplicated()].tolist()))
    if duplicate_ids:
        shown = ",".join(duplicate_ids[:10])
        suffix = "..." if len(duplicate_ids) > 10 else ""
        raise SystemExit(
            "ChEMBL fingerprint parquet contains duplicate molecule_chembl_id "
            f"values: {shown}{suffix}: {path}"
        )
    converted = []
    for molecule_id, bitvec in zip(
        normalized_ids,
        fp_df["bitvec"],
        strict=True,
    ):
        try:
            converted.append(fp_from_uint64_row(list(bitvec)))
        except (TypeError, ValueError, OverflowError) as exc:
            raise SystemExit(
                "ChEMBL fingerprint parquet contains invalid 2048-bit bitvec "
                f"for molecule {molecule_id}: {path}"
            ) from exc
    fp_df = fp_df.copy()
    fp_df["bitvec"] = converted
    return fp_df


def _parquet_columns(path: Path) -> set[str]:
    try:
        return set(pq.ParquetFile(path).schema.names)
    except Exception as exc:
        raise SystemExit(f"Unable to inspect ChEMBL human activities parquet: {path}") from exc


def load_activities(
    path: Path,
    *,
    quality_policy: str = "legacy",
    evidence_mode: str = "retrieval",
) -> pd.DataFrame:
    available = _parquet_columns(path)
    required = {"molecule_chembl_id", "uniprot"}
    if evidence_mode != "retrieval":
        required.add("standard_inchi_key")
    if quality_policy == "high-confidence":
        required |= HIGH_CONFIDENCE_COLUMNS
    elif quality_policy == "claim-grade":
        required |= CLAIM_GRADE_COLUMNS
    if evidence_mode == "temporal" and not ({"evidence_date", "document_year"} & available):
        required.add("evidence_date")
    missing = sorted(required - available)
    if missing:
        raise SystemExit(
            "ChEMBL human activities parquet missing required columns "
            f"{missing}: {path}"
        )
    optional = {
        "pchembl",
        "standard_relation",
        "assay_confidence_score",
        "potential_duplicate",
        "data_validity_comment",
        "evidence_date",
        "document_year",
        "activity_id",
        "assay_id",
        "document_id",
        "standard_inchi_key",
    }
    columns = sorted(required | (optional & available))
    try:
        act = pd.read_parquet(path, columns=columns)
    except Exception as exc:
        raise SystemExit(
            "Unable to read ChEMBL human activities parquet required columns "
            f"{columns}: {path}"
        ) from exc
    validate_nonblank_text_column(
        act, path, "ChEMBL human activities", "molecule_chembl_id"
    )
    validate_nonblank_text_column(act, path, "ChEMBL human activities", "uniprot")
    normalized = act.copy()
    normalized["molecule_chembl_id"] = (
        normalized["molecule_chembl_id"].astype(str).str.strip()
    )
    normalized["uniprot"] = normalized["uniprot"].astype(str).str.strip()
    if normalized.empty:
        raise SystemExit(
            f"ChEMBL human activities contain no molecule-target edges: {path}"
        )
    # Legacy evidence has no row-level provenance, so a repeated molecule-target
    # pair is the same edge twice and collapses. Rich evidence carries an
    # activity_id per row, so repeats are distinct measurements and stay.
    #
    # Both branches used to call a bare drop_duplicates() over every loaded
    # column, which removes nothing once activity_id is present - while the
    # warning above announced dropping 653,029 edges from the ChEMBL mirror. It
    # dropped none.
    legacy = set(normalized.columns) == {"molecule_chembl_id", "uniprot"}
    duplicate_count = int(
        normalized.duplicated(subset=["molecule_chembl_id", "uniprot"]).sum()
    )
    if legacy and duplicate_count:
        LOG.warning(
            "Dropping %d duplicate ChEMBL molecule-target activity edge(s) from %s",
            duplicate_count,
            path,
        )
        normalized = normalized.drop_duplicates().reset_index(drop=True)
    elif duplicate_count:
        LOG.info(
            "%d molecule-target pair(s) in %s carry more than one measurement; "
            "keeping them as distinct rows",
            duplicate_count,
            path,
        )
    return normalized


def _parse_cutoff(value: str | None) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise SystemExit(f"--cutoff-date must use YYYY-MM-DD: {value!r}") from exc


def _bool_series(series: pd.Series, *, column: str) -> pd.Series:
    def parse(value: object) -> bool:
        if pd.isna(value):
            return False
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)) and value in (0, 1):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"", "0", "false", "f", "no", "n"}:
            return False
        if text in {"1", "true", "t", "yes", "y"}:
            return True
        raise SystemExit(
            f"ChEMBL human activities column {column!r} contains invalid boolean "
            f"value {value!r}"
        )

    return series.map(parse).astype(bool)


def apply_quality_policy(act: pd.DataFrame, policy: str) -> tuple[pd.DataFrame, dict[str, int]]:
    if policy == "legacy":
        return act.copy(), {"before": len(act), "after": len(act)}
    required = (
        CLAIM_GRADE_COLUMNS if policy == "claim-grade" else HIGH_CONFIDENCE_COLUMNS
    )
    missing = sorted(required - set(act.columns))
    if missing:
        raise SystemExit(
            f"--quality-policy={policy} requires activity columns: "
            + ", ".join(missing)
        )
    relation = act["standard_relation"].fillna("").astype(str).str.strip()
    confidence = pd.to_numeric(act["assay_confidence_score"], errors="coerce")
    pchembl = pd.to_numeric(act["pchembl"], errors="coerce")
    duplicate = _bool_series(act["potential_duplicate"], column="potential_duplicate")
    validity = act["data_validity_comment"].fillna("").astype(str).str.strip()
    high_confidence_keep = (
        relation.eq("=")
        & confidence.ge(8.0)
        & pchembl.ge(5.0)
        & ~duplicate
        & validity.eq("")
    )
    if policy == "claim-grade":
        assay_type = act["assay_type"].fillna("").astype(str).str.strip().str.upper()
        target_type = act["target_type"].fillna("").astype(str).str.strip().str.upper()
        relationship = (
            act["assay_relationship_type"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )
        standard_type = (
            act["standard_type"].fillna("").astype(str).str.strip().str.upper()
        )
        standard_value = pd.to_numeric(act["standard_value"], errors="coerce")
        standard_units = (
            act["standard_units"].fillna("").astype(str).str.strip().str.lower()
        )
        document_type = (
            act["document_type"].fillna("").astype(str).str.strip().str.upper()
        )
        document_year = pd.to_numeric(act["document_year"], errors="coerce")
        assay_source = (
            act["assay_source_name"].fillna("").astype(str).str.strip().str.upper()
        )
        document_source = (
            act["document_source_name"].fillna("").astype(str).str.strip().str.upper()
        )
        validity_ok = validity.str.lower().isin({"", "manually validated"})
        direct_binding = (
            assay_type.eq("B")
            & target_type.eq("SINGLE PROTEIN")
            & confidence.eq(9.0)
            & relationship.eq("D")
        )
        quantitative = (
            relation.eq("=")
            & standard_type.isin({"IC50", "KI", "KD", "EC50"})
            & standard_units.eq("nm")
            & standard_value.gt(0.0)
            & pchembl.ge(5.0)
        )
        publication = document_type.eq("PUBLICATION") & document_year.notna()
        source_ok = ~assay_source.eq("BINDINGDB") & ~document_source.eq("BINDINGDB")
        wild_type = act["assay_variant_id"].isna()
        keep = (
            direct_binding
            & quantitative
            & publication
            & source_ok
            & wild_type
            & ~duplicate
            & validity_ok
        )
        stats = {
            "before": len(act),
            "after": int(keep.sum()),
            "rejected_not_direct_binding": int((~direct_binding).sum()),
            "rejected_not_quantitative_active": int((~quantitative).sum()),
            "rejected_not_dated_publication": int((~publication).sum()),
            "rejected_bindingdb_import": int((~source_ok).sum()),
            "rejected_variant": int((~wild_type).sum()),
            "rejected_duplicate": int(duplicate.sum()),
            "rejected_invalid_validity": int((~validity_ok).sum()),
        }
    else:
        keep = high_confidence_keep
        stats = {"before": len(act), "after": int(keep.sum())}
    filtered = act.loc[keep].copy()
    if filtered.empty:
        raise SystemExit(f"{policy} ChEMBL evidence policy removed every activity row")
    return filtered, stats


def apply_temporal_cutoff(
    act: pd.DataFrame,
    cutoff: date,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if "evidence_date" in act.columns:
        dates = pd.to_datetime(act["evidence_date"], errors="coerce").dt.date
    elif "document_year" in act.columns:
        years = pd.to_numeric(act["document_year"], errors="coerce")
        # ChEMBL exposes document year, not a trustworthy day. Treat year-end as
        # the evidence date so rows from the cutoff year are not leaked early.
        dates = years.map(
            lambda year: date(int(year), 12, 31) if pd.notna(year) else pd.NaT
        )
    else:
        raise SystemExit(
            "Temporal Daina mode requires evidence_date or document_year in "
            "human_activities.parquet"
        )
    dated = dates.notna()
    keep = dated & dates.map(lambda value: bool(value <= cutoff) if pd.notna(value) else False)
    filtered = act.loc[keep].copy()
    if filtered.empty:
        raise SystemExit(
            f"Temporal cutoff {cutoff.isoformat()} removed every dated activity row"
        )
    return filtered, {
        "before": len(act),
        "dated": int(dated.sum()),
        "undated_excluded": int((~dated).sum()),
        "after": len(filtered),
    }


def _potency_weight(series: pd.Series) -> pd.Series:
    pchembl = pd.to_numeric(series, errors="coerce")
    scaled = ((pchembl - 5.0) / 3.0).clip(lower=0.0, upper=1.0)
    return (0.5 + 0.5 * scaled).fillna(0.5)


def aggregate_target_scores(act: pd.DataFrame, scoring_method: str) -> pd.DataFrame:
    act = act.copy()
    grouped = act.groupby("uniprot", sort=True)
    per_target = grouped["sim"].max().rename("max_tanimoto").to_frame()
    # Distinct ligands, not activity rows. Downstream renders this as
    # "이 표적의 측정된 리간드 수" and the row count is a different quantity:
    # P00352 in one real run reported 76,325 against 75,588 distinct molecules.
    per_target["evidence_count"] = grouped["molecule_chembl_id"].nunique().astype(int)
    best_indexes = grouped["sim"].idxmax()
    per_target["supporting_molecule_id"] = [
        str(act.loc[idx, "molecule_chembl_id"]) for idx in best_indexes
    ]
    if scoring_method == "quality-hybrid":
        if "pchembl" not in act.columns:
            raise SystemExit(
                "--scoring-method=quality-hybrid requires pchembl in "
                "human_activities.parquet"
            )
        act["quality_similarity"] = act["sim"] * _potency_weight(act["pchembl"])
        quality = act.groupby("uniprot", sort=True)["quality_similarity"].max()
        per_target["max_quality_similarity"] = quality
        per_target["score"] = 0.7 * per_target["max_tanimoto"] + 0.3 * quality
    else:
        per_target["score"] = per_target["max_tanimoto"]
    per_target = per_target.reset_index().rename(columns={"uniprot": "target_id"})
    return per_target.sort_values(
        ["score", "target_id"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)


def validate_reference_overlap(act: pd.DataFrame, fp_df: pd.DataFrame, activities_path: Path) -> None:
    fingerprint_ids = set(fp_df["molecule_chembl_id"].astype(str))
    matched = act["molecule_chembl_id"].isin(fingerprint_ids)
    if not matched.any():
        raise SystemExit(
            "Daina-Zoete has no overlapping molecule IDs between fingerprints "
            f"and human activities: {activities_path}"
        )
    missing = int((~matched).sum())
    if missing:
        missing_ids = sorted(set(act.loc[~matched, "molecule_chembl_id"].astype(str)))
        shown = ",".join(missing_ids[:10])
        suffix = "..." if len(missing_ids) > 10 else ""
        raise SystemExit(
            "Daina-Zoete refuses partial ChEMBL evidence: "
            f"{missing} activity edges without fingerprints "
            f"({shown}{suffix}) in {activities_path}; rebuild the fingerprint "
            "parquet from the same human_activities.parquet."
        )


def score_query_against_reference(
    *,
    qfp: "DataStructs.ExplicitBitVect",
    query_connectivity_key: str | None,
    fp_df: pd.DataFrame,
    act: pd.DataFrame,
    evidence_mode: str,
    exclude_reference_similarity: float,
    scoring_method: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    similarities = DataStructs.BulkTanimotoSimilarity(
        qfp,
        fp_df["bitvec"].tolist(),
    )
    sim_lookup = dict(
        zip(
            fp_df["molecule_chembl_id"].astype(str),
            (float(value) for value in similarities),
            strict=True,
        )
    )
    scored = act.copy()
    scored["sim"] = scored["molecule_chembl_id"].map(sim_lookup)
    if scored["sim"].isna().any():
        raise SystemExit(
            "Daina-Zoete encountered activity rows without validated fingerprints"
        )
    excluded_similarity_reference_molecules = 0
    excluded_similarity_activity_rows = 0
    excluded_identity_activity_rows = 0
    excluded_identity_reference_molecules = 0
    excluded_total_activity_rows = 0
    excluded_total_reference_molecules = 0
    if evidence_mode != "retrieval":
        if not query_connectivity_key:
            raise SystemExit(
                "Leakage-controlled Daina scoring requires a query InChIKey "
                "connectivity block"
            )
        if "standard_inchi_key" not in scored.columns:
            raise SystemExit(
                "Leakage-controlled Daina scoring requires standard_inchi_key "
                "in human_activities.parquet"
            )
        reference_keys = (
            scored["standard_inchi_key"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
            .str.split("-", n=1)
            .str[0]
        )
        identity_excluded = reference_keys.eq(query_connectivity_key)
        excluded_identity_activity_rows = int(identity_excluded.sum())
        excluded_identity_reference_molecules = int(
            scored.loc[identity_excluded, "molecule_chembl_id"].nunique()
        )
        similarity_excluded = scored["sim"].ge(exclude_reference_similarity)
        excluded_similarity_activity_rows = int(similarity_excluded.sum())
        excluded_similarity_reference_molecules = int(
            scored.loc[similarity_excluded, "molecule_chembl_id"].nunique()
        )
        excluded = similarity_excluded | identity_excluded
        excluded_total_activity_rows = int(excluded.sum())
        excluded_total_reference_molecules = int(
            scored.loc[excluded, "molecule_chembl_id"].nunique()
        )
        scored = scored.loc[~excluded].copy()
        if scored.empty:
            raise SystemExit(
                "Leakage-controlled Daina identity/similarity filtering removed "
                "every reference molecule at similarity threshold "
                f">= {exclude_reference_similarity:g}"
            )
    per_target = aggregate_target_scores(scored, scoring_method)
    per_target["evidence_mode"] = evidence_mode
    per_target["scoring_method"] = scoring_method
    return per_target, {
        "pre_similarity_filter_activity_rows": len(act),
        "scored_activity_rows": len(scored),
        "excluded_activity_rows_by_similarity": excluded_similarity_activity_rows,
        "excluded_reference_molecules_by_similarity": (
            excluded_similarity_reference_molecules
        ),
        "excluded_activity_rows_by_identity": excluded_identity_activity_rows,
        "excluded_reference_molecules_by_identity": excluded_identity_reference_molecules,
        "excluded_activity_rows_total": excluded_total_activity_rows,
        "excluded_reference_molecules_total": excluded_total_reference_molecules,
    }


def _score_with_recipe(
    args: argparse.Namespace, qfp: "DataStructs.ExplicitBitVect"
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Rank with the promoted recipe against the retrieval index.

    The recipe weights evidence the nearest-neighbour score cannot see. For the
    promoted `union_any_consensus` that is: similarity to the nearest measured
    analogue at any potency (0.45), the same restricted to analogues at
    pActivity >= 6 (0.30), a potency-weighted variant (0.15), and whether a
    second source database also has something similar (0.10). Publication counts
    are built into the index but no feature reads them, and this recipe gives
    zero weight to support and negative-contrast.

    It reads the index, not the mirror, so the index's role matters: an
    evaluation index is built from the benchmark's train split and would show a
    researcher less than the mirror does.
    """
    from stage3_recipe_scoring import load_operational_recipe, score_query_with_recipe

    index_dir = args.recipe_index_dir
    manifest_path = index_dir / "manifest.json"
    for required in (index_dir / "ligands.parquet", index_dir / "edges.parquet", manifest_path):
        if not required.exists():
            raise SystemExit(f"retrieval index is incomplete, missing: {required}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("index_role") != "production":
        raise SystemExit(
            f"--recipe-index-dir must be a production index, not "
            f"{manifest.get('index_role', 'evaluation')!r}: {manifest_path}. An "
            "evaluation index is train-only and would narrow what a run can find."
        )

    cluster_manifest = args.target_clusters.with_suffix("").with_suffix(".manifest.json")
    if not cluster_manifest.exists():
        cluster_manifest = args.target_clusters.parent / (
            args.target_clusters.stem + ".manifest.json"
        )

    if args.operational_gate is None:
        raise SystemExit("--operational-gate is required when --recipe is used")
    gate = json.loads(args.operational_gate.read_text(encoding="utf-8"))
    if gate.get("schema_version") != "skinscout.activity-retrieval-operational-gate.v1":
        raise SystemExit(f"invalid activity retrieval operational gate: {args.operational_gate}")
    if gate.get("runtime_recipe", {}).get("sha256") != _sha256(args.recipe):
        raise SystemExit("operational gate does not bind the configured runtime recipe")
    if gate.get("runtime_index_manifest", {}).get("sha256") != _sha256(manifest_path):
        raise SystemExit("operational gate does not bind the configured runtime index")
    operational_recipe_id = str(gate.get("operational_recipe_id") or "").strip()
    if not operational_recipe_id:
        raise SystemExit("operational gate lacks operational_recipe_id")
    recipe = load_operational_recipe(args.recipe, operational_recipe_id)
    per_target, stats = score_query_with_recipe(
        query_fp=qfp,
        recipe=recipe,
        ligands_path=index_dir / "ligands.parquet",
        edges_path=index_dir / "edges.parquet",
        index_manifest_path=manifest_path,
        target_csv=args.target_clusters,
        target_manifest_path=cluster_manifest,
        # `retrieval` keeps the compound's own measurements in view: a researcher
        # checking a known ingredient should be told it is known. The leakage
        # modes exist for measurement, where that would be answer lookup.
        #
        # One difference from the mirror path, stated rather than hidden: this
        # excludes by similarity only, while score_query_against_reference also
        # drops references whose InChIKey connectivity block equals the query's.
        # Fingerprints ignore chirality, so a stereoisomer already sits at 1.0
        # and the threshold catches it; every panel measurement in
        # docs/RECIPE_RUNPATH_MEASURED_20260831.md was made this way. Production
        # runs use `retrieval`, where neither exclusion applies.
        exclude_reference_similarity=(
            None if args.evidence_mode == "retrieval" else args.exclude_reference_similarity
        ),
    )
    if per_target.empty:
        raise SystemExit("recipe scoring produced no target scores")
    per_target["evidence_mode"] = args.evidence_mode
    per_target["scoring_method"] = "recipe:" + recipe.recipe_id
    provenance = {
        "recipe": str(args.recipe),
        "recipe_sha256": _sha256(args.recipe),
        "index_dir": str(index_dir),
        "index_manifest_sha256": _sha256(manifest_path),
        "index_ligands_sha256": manifest.get("outputs", {}).get("ligands", {}).get("sha256"),
        "index_edges_sha256": manifest.get("outputs", {}).get("edges", {}).get("sha256"),
        "evidence_licensing": manifest.get("evidence_licensing"),
    }
    return per_target, stats, provenance


def _recipe_metadata(
    args: argparse.Namespace,
    cutoff: date | None,
    stats: dict[str, Any],
    provenance: dict[str, Any],
    per_target: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "schema_version": "skinscout.daina-run.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_mode": args.evidence_mode,
        "evidence_snapshot_id": args.evidence_snapshot_id,
        "cutoff_date": cutoff.isoformat() if cutoff is not None else None,
        "exclude_reference_similarity": stats.get("exclude_reference_similarity"),
        # Guarded above to be exactly these, so the record is honest rather than
        # an echo of an argument nothing read.
        "quality_policy": "index-label-policy",
        "scoring_method": "recipe:" + str(stats["recipe_id"]),
        "score_is_calibrated_probability": False,
        "inputs": {
            "ligand_sdf": str(args.ligand_sdf),
            "ligand_sdf_sha256": _sha256(args.ligand_sdf),
            **provenance,
        },
        "counts": {**stats, "ranked_targets": len(per_target)},
        # The mirror's quality and temporal filters do not run on this path: the
        # index applied its own label policy when it was built.
        "quality_filter": None,
        "temporal_filter": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ligand-sdf", required=True, type=Path)
    parser.add_argument(
        "--chembl-fp",
        type=Path,
        help="ChEMBL fingerprint parquet; required unless --recipe is used",
    )
    parser.add_argument("--out-scores", required=True, type=Path)
    parser.add_argument("--out-metadata-json", type=Path)
    parser.add_argument(
        "--evidence-mode",
        choices=EVIDENCE_MODES,
        default="retrieval",
        help=(
            "retrieval preserves the historical lookup baseline; "
            "leave-query-out and temporal exclude high-similarity references"
        ),
    )
    parser.add_argument(
        "--evidence-snapshot-id",
        help="Immutable source manifest id or digest used by leakage-controlled modes.",
    )
    parser.add_argument(
        "--cutoff-date",
        help="Latest evidence date allowed in temporal mode (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--exclude-reference-similarity",
        type=float,
        default=0.85,
        help=(
            "In non-retrieval modes, exclude reference molecules with query "
            "Tanimoto at or above this value."
        ),
    )
    parser.add_argument(
        "--quality-policy",
        choices=QUALITY_POLICIES,
        default="legacy",
    )
    parser.add_argument(
        "--scoring-method",
        choices=SCORING_METHODS,
        default="max-similarity",
    )
    parser.add_argument(
        "--recipe",
        type=Path,
        help=(
            "promoted retrieval recipe. Given together with --recipe-index-dir "
            "this replaces nearest-neighbour ranking with the weighted recipe "
            "score, evaluated against the retrieval index rather than the mirror"
        ),
    )
    parser.add_argument("--recipe-index-dir", type=Path)
    parser.add_argument("--operational-gate", type=Path)
    parser.add_argument(
        "--target-clusters",
        type=Path,
        default=ROOT / "data" / "evidence_splits" / "screenable_target_clusters_2026_02.csv",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    outputs = [args.out_scores]
    if args.out_metadata_json is not None:
        outputs.append(args.out_metadata_json)
    _remove_outputs(*outputs)
    if not math.isfinite(args.exclude_reference_similarity) or not (
        0.0 < args.exclude_reference_similarity <= 1.0
    ):
        raise SystemExit("--exclude-reference-similarity must be finite and in (0, 1]")
    if (args.recipe is None) != (args.recipe_index_dir is None):
        raise SystemExit("--recipe and --recipe-index-dir must be given together")
    if args.recipe is None and args.chembl_fp is None:
        raise SystemExit("--chembl-fp is required unless --recipe is used")
    if args.recipe is not None and args.scoring_method != "max-similarity":
        raise SystemExit(
            "--recipe replaces the aggregation entirely; "
            f"--scoring-method={args.scoring_method} would be ignored"
        )
    # The recipe scores against a prebuilt index, so the mirror's row-level
    # filters have nothing to act on. Accepting them and recording them in the
    # metadata would let a run claim a quality policy or a date holdout it never
    # applied - and that metadata is the only record either one ever gets.
    if args.recipe is not None and args.quality_policy != "legacy":
        raise SystemExit(
            f"--quality-policy={args.quality_policy} cannot be applied on the "
            "recipe path: the index carries its own label policy, fixed when it "
            "was built. Rebuild the index under the policy you want instead."
        )
    if args.recipe is not None and args.evidence_mode == "temporal":
        raise SystemExit(
            "--evidence-mode=temporal cannot be applied on the recipe path: the "
            "production index carries no dates, so a cutoff would filter nothing "
            "while the metadata recorded one. Score against the mirror for a "
            "date-controlled run."
        )
    cutoff = _parse_cutoff(args.cutoff_date)
    if args.evidence_mode == "temporal" and cutoff is None:
        raise SystemExit("--evidence-mode=temporal requires --cutoff-date")
    if args.evidence_mode != "temporal" and cutoff is not None:
        raise SystemExit("--cutoff-date is only valid with --evidence-mode=temporal")
    if args.evidence_mode != "retrieval" and not str(
        args.evidence_snapshot_id or ""
    ).strip():
        raise SystemExit(
            f"--evidence-mode={args.evidence_mode} requires --evidence-snapshot-id"
        )
    if args.evidence_mode != "retrieval" and args.out_metadata_json is None:
        raise SystemExit(
            f"--evidence-mode={args.evidence_mode} requires --out-metadata-json"
        )
    qfp, query_connectivity_key = query_features(args.ligand_sdf)
    if args.recipe is not None:
        per_target, scoring_stats, recipe_provenance = _score_with_recipe(args, qfp)
        per_target["quality_policy"] = args.quality_policy
        _write_tsv_atomic(per_target, args.out_scores)
        if args.out_metadata_json is not None:
            _write_json_atomic(
                _recipe_metadata(args, cutoff, scoring_stats, recipe_provenance, per_target),
                args.out_metadata_json,
            )
        LOG.info(
            "Daina-Zoete recipe scores -> %s (%d targets, recipe=%s)",
            args.out_scores,
            len(per_target),
            scoring_stats["recipe_id"],
        )
        return

    if args.chembl_fp is None or not args.chembl_fp.exists():
        raise SystemExit(
            f"ChEMBL fingerprint parquet is required for Daina-Zoete scoring: {args.chembl_fp}"
        )

    fp_df = load_fingerprints(args.chembl_fp)

    # Need (molecule, uniprot) edges; reload from human_activities.parquet
    activities_path = args.chembl_fp.parent / "human_activities.parquet"
    if not activities_path.exists():
        raise SystemExit(f"Missing {activities_path}")
    act = load_activities(
        activities_path,
        quality_policy=args.quality_policy,
        evidence_mode=args.evidence_mode,
    )
    initial_activity_rows = len(act)
    act, quality_stats = apply_quality_policy(act, args.quality_policy)
    temporal_stats: dict[str, int] | None = None
    if cutoff is not None:
        act, temporal_stats = apply_temporal_cutoff(act, cutoff)

    validate_reference_overlap(act, fp_df, activities_path)
    per_target, scoring_stats = score_query_against_reference(
        qfp=qfp,
        query_connectivity_key=query_connectivity_key,
        fp_df=fp_df,
        act=act,
        evidence_mode=args.evidence_mode,
        exclude_reference_similarity=args.exclude_reference_similarity,
        scoring_method=args.scoring_method,
    )
    per_target["quality_policy"] = args.quality_policy
    if per_target.empty:
        raise SystemExit("Daina-Zoete produced no target scores")

    _write_tsv_atomic(per_target, args.out_scores)
    if args.out_metadata_json is not None:
        metadata = {
            "schema_version": "skinscout.daina-run.v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "evidence_mode": args.evidence_mode,
            "evidence_snapshot_id": args.evidence_snapshot_id,
            "cutoff_date": cutoff.isoformat() if cutoff is not None else None,
            "exclude_reference_similarity": (
                args.exclude_reference_similarity
                if args.evidence_mode != "retrieval"
                else None
            ),
            "quality_policy": args.quality_policy,
            "scoring_method": args.scoring_method,
            "score_is_calibrated_probability": False,
            "inputs": {
                "ligand_sdf": str(args.ligand_sdf),
                "ligand_sdf_sha256": _sha256(args.ligand_sdf),
                "fingerprints": str(args.chembl_fp),
                "fingerprints_sha256": _sha256(args.chembl_fp),
                "activities": str(activities_path),
                "activities_sha256": _sha256(activities_path),
            },
            "counts": {
                "initial_activity_rows": initial_activity_rows,
                **scoring_stats,
                "ranked_targets": len(per_target),
            },
            "quality_filter": quality_stats,
            "temporal_filter": temporal_stats,
        }
        _write_json_atomic(metadata, args.out_metadata_json)
    LOG.info("Daina-Zoete scores → %s (%d targets)", args.out_scores, len(per_target))


if __name__ == "__main__":
    main()
