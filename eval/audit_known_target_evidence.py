#!/usr/bin/env python3
"""Retrospective ground-truth provenance audit for known-target panel pairs.

This audit is output-only evidence accounting. It must never be used as ranking
input, training input, candidate generation input, or any other predictive
feature source.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


SCHEMA_VERSION = "known_target_provenance_audit.v1"
AUDIT_LABEL = "retrospective_ground_truth_provenance_audit"
SEPARATION_LABEL = "audit_only_never_ranking_input"
DEFAULT_PANEL = Path("data/validation/skin_known_target_panel.csv")
DEFAULT_EVIDENCE = Path("data/chembl37/activity_evidence.parquet")
DEFAULT_SOURCE_MANIFEST = Path("data/chembl37/source_manifest.json")
DEFAULT_OUT_CSV = Path("results/audits/known_target_pair_evidence.csv")
DEFAULT_OUT_MANIFEST = Path("results/audits/known_target_pair_evidence.manifest.json")
ENDPOINTS = {"IC50", "KI", "KD", "EC50"}
CHEMBL_REQUIRED_RELEASE = "37"
VALIDITY_COMMENTS = {"", "manually validated"}
INCHIKEY_RE = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
REQUIRED_PANEL_COLUMNS = {"case_id", "smiles", "known_targets"}
REQUIRED_EVIDENCE_COLUMNS = {
    "standard_inchi_key",
    "uniprot",
    "assay_type",
    "target_type",
    "assay_confidence_score",
    "assay_relationship_type",
    "standard_relation",
    "standard_type",
    "standard_value",
    "standard_units",
    "data_validity_comment",
    "potential_duplicate",
    "document_type",
    "document_year",
    "assay_variant_id",
    "assay_variant_accession",
    "assay_variant_mutation",
    "assay_source_name",
    "document_source_name",
    "pchembl_value",
    "activity_id",
    "document_id",
    "document_chembl_id",
    "pubmed_id",
    "doi",
}
PAIR_COLUMNS = [
    "audit_label",
    "separation_label",
    "case_id",
    "panel_smiles",
    "parent_inchikey",
    "parent_inchikey_connectivity",
    "uniprot",
    "known_target_label",
    "support_status",
    "support_count",
    "best_pchembl",
    "earliest_year",
    "latest_year",
    "sample_dois",
    "sample_pmids",
    "sample_document_ids",
    "sample_document_chembl_ids",
    "sample_activity_ids",
]
SAMPLE_LIMIT = 25


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


def _write_csv_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def parent_inchikey_connectivity(smiles: str) -> tuple[str, str]:
    text = _clean(smiles)
    if not text:
        raise SystemExit("Known-target panel contains blank SMILES")
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        raise SystemExit(f"Known-target panel contains invalid SMILES: {text!r}")
    parent = rdMolStandardize.FragmentParent(mol)
    inchikey = Chem.MolToInchiKey(parent)
    if not INCHIKEY_RE.fullmatch(inchikey):
        raise SystemExit(f"RDKit produced malformed parent InChIKey for SMILES {text!r}: {inchikey!r}")
    return inchikey, inchikey.split("-", 1)[0]


def inchikey_connectivity(value: object) -> str:
    text = _clean(value).upper()
    if not text:
        return ""
    if not INCHIKEY_RE.fullmatch(text):
        raise SystemExit(f"ChEMBL evidence contains malformed standard_inchi_key: {text!r}")
    return text.split("-", 1)[0]


def _split_semicolon(value: object, label: str, row_id: str) -> list[str]:
    text = _clean(value)
    if not text:
        raise SystemExit(f"Known-target panel column '{label}' contains blank values: {row_id}")
    tokens = [token.strip() for token in text.split(";")]
    if any(not token for token in tokens):
        raise SystemExit(
            f"Known-target panel column '{label}' contains empty ';'-separated token: {row_id}"
        )
    return tokens


def _read_panel(path: Path) -> tuple[list[dict[str, object]], dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"Known-target panel is required and must be non-empty: {path}")
    try:
        panel = pd.read_csv(path)
    except Exception as exc:
        raise SystemExit(f"Known-target panel failed to parse: {path}: {exc}") from exc
    missing = sorted(REQUIRED_PANEL_COLUMNS - set(panel.columns))
    if missing:
        raise SystemExit(f"Known-target panel missing required columns {', '.join(missing)}: {path}")
    if panel.empty:
        raise SystemExit(f"Known-target panel contains no rows: {path}")

    pairs: list[dict[str, object]] = []
    seen_case_ids: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    has_labels = "known_target_labels" in panel.columns
    for idx, row in panel.iterrows():
        case_id = _clean(row["case_id"])
        if not case_id:
            raise SystemExit(f"Known-target panel contains blank case_id at row {idx}")
        if case_id in seen_case_ids:
            raise SystemExit(f"Known-target panel contains duplicate case_id: {case_id}")
        seen_case_ids.add(case_id)
        smiles = _clean(row["smiles"])
        parent_key, connectivity = parent_inchikey_connectivity(smiles)
        targets = _split_semicolon(row["known_targets"], "known_targets", f"case_id={case_id}")
        labels = (
            _split_semicolon(row["known_target_labels"], "known_target_labels", f"case_id={case_id}")
            if has_labels and _clean(row.get("known_target_labels"))
            else ["" for _ in targets]
        )
        if len(labels) != len(targets):
            raise SystemExit(f"known_target_labels count must match known_targets count: case_id={case_id}")
        for uniprot, label in zip(targets, labels, strict=True):
            uniprot = _clean(uniprot)
            if not uniprot:
                raise SystemExit(f"Known-target panel contains blank UniProt target: case_id={case_id}")
            pair_key = (case_id, uniprot)
            if pair_key in seen_pairs:
                raise SystemExit(f"Known-target panel contains duplicate pair: {case_id}/{uniprot}")
            seen_pairs.add(pair_key)
            pairs.append(
                {
                    "case_id": case_id,
                    "panel_smiles": smiles,
                    "parent_inchikey": parent_key,
                    "parent_inchikey_connectivity": connectivity,
                    "uniprot": uniprot,
                    "known_target_label": _clean(label),
                }
            )
    return pairs, {"path": str(path), "sha256": _sha256_file(path), "rows": len(panel), "pairs": len(pairs)}


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"ChEMBL source manifest is required and must be non-empty: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"ChEMBL source manifest is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"ChEMBL source manifest must be a JSON object: {path}")
    return payload


def _validate_source_manifest(
    path: Path, evidence_meta: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _read_json_object(path)
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise SystemExit(f"ChEMBL source manifest missing object field 'source': {path}")
    release = _clean(source.get("release"))
    license_text = _clean(source.get("license"))
    if _clean(source.get("name")).casefold() != "chembl":
        raise SystemExit(f"ChEMBL source manifest has unexpected source name: {path}")
    if release != CHEMBL_REQUIRED_RELEASE:
        raise SystemExit(
            f"ChEMBL source manifest release must be {CHEMBL_REQUIRED_RELEASE}: {path}"
        )
    if not license_text:
        raise SystemExit(f"ChEMBL source manifest missing source release/license: {path}")
    output_hashes = manifest.get("output_sha256")
    if not isinstance(output_hashes, dict) or evidence_meta["sha256"] not in {
        _clean(value) for value in output_hashes.values()
    }:
        raise SystemExit(
            "ChEMBL source manifest output hashes do not bind the current evidence parquet: "
            f"{path}"
        )
    row_counts = manifest.get("row_counts")
    if (
        not isinstance(row_counts, dict)
        or row_counts.get("activity_evidence") != evidence_meta["rows"]
    ):
        raise SystemExit(
            "ChEMBL source manifest activity_evidence row count does not match the current "
            f"parquet: {path}"
        )
    return manifest, {
        "path": str(path),
        "sha256": _sha256_file(path),
        "release": release,
        "license": license_text,
        "schema_version": _clean(manifest.get("schema_version")),
    }


def _validate_evidence_schema(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"ChEMBL activity evidence parquet is required and must be non-empty: {path}")
    try:
        parquet = pq.ParquetFile(path)
    except Exception as exc:
        raise SystemExit(f"ChEMBL activity evidence parquet failed to open: {path}: {exc}") from exc
    columns = set(parquet.schema_arrow.names)
    missing = sorted(REQUIRED_EVIDENCE_COLUMNS - columns)
    if missing:
        raise SystemExit(f"ChEMBL activity evidence missing required columns {', '.join(missing)}: {path}")
    return {"path": str(path), "sha256": _sha256_file(path), "rows": parquet.metadata.num_rows}


def _bool_false(value: object) -> bool:
    return _clean(value).casefold() in {"", "0", "false", "f", "no", "n"}


def _validity_ok(value: object) -> bool:
    return _clean(value).casefold() in VALIDITY_COMMENTS


def _valid_year(value: object) -> int | None:
    text = _clean(value)
    if not text:
        return None
    try:
        year = int(float(text))
    except ValueError:
        return None
    if 1800 <= year <= 2100:
        return year
    return None


def _is_variant(row: pd.Series) -> bool:
    variant_id = _clean(row.get("assay_variant_id"))
    if variant_id:
        return True
    return bool(_clean(row.get("assay_variant_accession")) or _clean(row.get("assay_variant_mutation")))


def _is_bindingdb_source(row: pd.Series) -> bool:
    names = [
        _clean(row.get("assay_source_name")).casefold(),
        _clean(row.get("document_source_name")).casefold(),
    ]
    return any("bindingdb" in name for name in names)


def passes_claim_grade_filters(row: pd.Series) -> bool:
    endpoint = _clean(row.get("standard_type")).upper()
    try:
        value = float(row.get("standard_value"))
    except (TypeError, ValueError):
        value = math.nan
    try:
        pchembl = float(row.get("pchembl_value"))
    except (TypeError, ValueError):
        pchembl = math.nan
    return (
        _clean(row.get("assay_type")) == "B"
        and _clean(row.get("target_type")) == "SINGLE PROTEIN"
        and _clean(row.get("assay_confidence_score")) == "9"
        and _clean(row.get("assay_relationship_type")) == "D"
        and _clean(row.get("standard_relation")) == "="
        and _clean(row.get("standard_units")).casefold() == "nm"
        and endpoint in ENDPOINTS
        and math.isfinite(value)
        and value > 0.0
        and math.isfinite(pchembl)
        and pchembl >= 5.0
        and _validity_ok(row.get("data_validity_comment"))
        and _bool_false(row.get("potential_duplicate"))
        and _clean(row.get("document_type")).casefold() == "publication"
        and _valid_year(row.get("document_year")) is not None
        and not _is_variant(row)
        and not _is_bindingdb_source(row)
    )


def _append_sample(bucket: set[str], value: object) -> None:
    text = _clean(value)
    if text and len(bucket) < SAMPLE_LIMIT:
        bucket.add(text)


def _scan_evidence(path: Path, wanted_pairs: set[tuple[str, str]], columns: Iterable[str]) -> dict[tuple[str, str], dict[str, Any]]:
    support: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "support_count": 0,
            "best_pchembl": None,
            "earliest_year": None,
            "latest_year": None,
            "dois": set(),
            "pmids": set(),
            "document_ids": set(),
            "document_chembl_ids": set(),
            "activity_ids": set(),
        }
    )
    dataset = ds.dataset(path, format="parquet")
    for batch in dataset.to_batches(columns=list(columns)):
        frame = batch.to_pandas()
        for _, row in frame.iterrows():
            key = (inchikey_connectivity(row.get("standard_inchi_key")), _clean(row.get("uniprot")))
            if key not in wanted_pairs:
                continue
            if not passes_claim_grade_filters(row):
                continue
            item = support[key]
            item["support_count"] += 1
            pchembl = pd.to_numeric(row.get("pchembl_value"), errors="coerce")
            if not pd.isna(pchembl):
                item["best_pchembl"] = max(
                    float(pchembl),
                    item["best_pchembl"] if item["best_pchembl"] is not None else float("-inf"),
                )
            year = _valid_year(row.get("document_year"))
            if year is not None:
                item["earliest_year"] = (
                    year if item["earliest_year"] is None else min(int(item["earliest_year"]), year)
                )
                item["latest_year"] = (
                    year if item["latest_year"] is None else max(int(item["latest_year"]), year)
                )
            _append_sample(item["dois"], row.get("doi"))
            _append_sample(item["pmids"], row.get("pubmed_id"))
            _append_sample(item["document_ids"], row.get("document_id"))
            _append_sample(item["document_chembl_ids"], row.get("document_chembl_id"))
            _append_sample(item["activity_ids"], row.get("activity_id"))
    return support


def _sample_text(values: set[str]) -> str:
    return ";".join(sorted(values))


def _pair_rows(pairs: list[dict[str, object]], support: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pair in pairs:
        key = (
            str(pair["parent_inchikey_connectivity"]),
            str(pair["uniprot"]),
        )
        item = support.get(key)
        supported = item is not None and int(item["support_count"]) > 0
        rows.append(
            {
                "audit_label": AUDIT_LABEL,
                "separation_label": SEPARATION_LABEL,
                **pair,
                "support_status": "supported" if supported else "unsupported",
                "support_count": int(item["support_count"]) if supported else 0,
                "best_pchembl": "" if not supported or item["best_pchembl"] is None else item["best_pchembl"],
                "earliest_year": "" if not supported or item["earliest_year"] is None else item["earliest_year"],
                "latest_year": "" if not supported or item["latest_year"] is None else item["latest_year"],
                "sample_dois": "" if not supported else _sample_text(item["dois"]),
                "sample_pmids": "" if not supported else _sample_text(item["pmids"]),
                "sample_document_ids": "" if not supported else _sample_text(item["document_ids"]),
                "sample_document_chembl_ids": "" if not supported else _sample_text(item["document_chembl_ids"]),
                "sample_activity_ids": "" if not supported else _sample_text(item["activity_ids"]),
            }
        )
    return rows


def run_audit(
    *,
    panel_csv: Path,
    evidence_parquet: Path,
    source_manifest: Path,
    out_csv: Path,
    out_manifest: Path,
) -> dict[str, Any]:
    _remove_outputs(out_csv, out_manifest)
    try:
        pairs, panel_meta = _read_panel(panel_csv)
        evidence_meta = _validate_evidence_schema(evidence_parquet)
        _, source_meta = _validate_source_manifest(source_manifest, evidence_meta)
        wanted_pairs = {(str(pair["parent_inchikey_connectivity"]), str(pair["uniprot"])) for pair in pairs}
        support = _scan_evidence(
            evidence_parquet,
            wanted_pairs,
            REQUIRED_EVIDENCE_COLUMNS,
        )
        rows = _pair_rows(pairs, support)
        _write_csv_atomic(out_csv, rows)
        pair_counts = {
            "supported": sum(1 for row in rows if row["support_status"] == "supported"),
            "unsupported": sum(1 for row in rows if row["support_status"] == "unsupported"),
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "audit_label": AUDIT_LABEL,
            "separation_label": SEPARATION_LABEL,
            "separation_policy": (
                "Retrospective ground-truth provenance audit only; outputs are never "
                "ranking input, training input, prediction input, or candidate-generation input."
            ),
            "filters": {
                "assay_type": "B",
                "target_type": "SINGLE PROTEIN",
                "assay_confidence_score": 9,
                "assay_relationship_type": "D",
                "standard_relation": "=",
                "standard_units": "nM",
                "standard_type": sorted(ENDPOINTS),
                "standard_value": ">0",
                "pchembl_value": ">=5 and finite",
                "data_validity_comment": sorted(VALIDITY_COMMENTS),
                "potential_duplicate": "false/0/blank only",
                "document_type": "publication",
                "document_year": "valid year required",
                "assay_variant": "excluded",
                "source_name": "exclude BindingDB by assay/document source name",
            },
            "inputs": {
                "panel": panel_meta,
                "evidence": evidence_meta,
                "source_manifest": source_meta,
            },
            "outputs": {
                "pair_csv": {"path": str(out_csv), "sha256": _sha256_file(out_csv), "rows": len(rows)},
            },
            "source": {
                "name": "ChEMBL",
                "release": source_meta["release"],
                "license": source_meta["license"],
            },
            "pair_counts": {
                "total": len(rows),
                **pair_counts,
            },
            "per_pair": [
                {
                    "case_id": row["case_id"],
                    "uniprot": row["uniprot"],
                    "support_status": row["support_status"],
                    "support_count": row["support_count"],
                    "best_pchembl": row["best_pchembl"],
                    "earliest_year": row["earliest_year"],
                    "latest_year": row["latest_year"],
                    "sample_activity_ids": row["sample_activity_ids"],
                }
                for row in rows
            ],
            "sample_limit_per_identifier_field": SAMPLE_LIMIT,
            "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        _write_json_atomic(out_manifest, manifest)
        return manifest
    except Exception:
        _remove_outputs(out_csv, out_manifest)
        raise


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-csv", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--evidence-parquet", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-manifest", type=Path, default=DEFAULT_OUT_MANIFEST)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        manifest = run_audit(
            panel_csv=args.panel_csv,
            evidence_parquet=args.evidence_parquet,
            source_manifest=args.source_manifest,
            out_csv=args.out_csv,
            out_manifest=args.out_manifest,
        )
    except Exception as exc:
        print(f"[known-target-audit][FATAL] {exc}", file=sys.stderr)
        return 1
    counts = manifest["pair_counts"]
    print(
        "[known-target-audit] wrote retrospective audit pairs="
        f"{counts['total']} supported={counts['supported']} unsupported={counts['unsupported']} "
        f"csv={args.out_csv} manifest={args.out_manifest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
