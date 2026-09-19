#!/usr/bin/env python3
"""Build generic temporal/cold evidence splits from normalized activity evidence."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import shutil
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


SCHEMA_VERSION = "evidence_temporal_cold_splits.v1"
OUTPUT_FILES = {
    "pre_cutoff": "pre_cutoff.parquet",
    "post_cutoff": "post_cutoff.parquet",
    "undated": "undated.parquet",
    "post_cutoff_novel_pairs": "post_cutoff_novel_pairs.parquet",
}
MANIFEST_NAME = "split_manifest.json"

SOURCE_DB_COLUMNS = ("source_db", "source_name", "database", "source")
SOURCE_RELEASE_COLUMNS = ("source_release", "release", "version", "source_version")
SOURCE_LICENSE_COLUMNS = ("source_license", "license")
TARGET_COLUMNS = ("target_uniprot", "uniprot", "accession", "target_accession")
MOLECULE_ID_COLUMNS = ("molecule_id", "ligand_id", "molecule_chembl_id", "compound_id")
SMILES_COLUMNS = ("molecule_smiles", "smiles", "ligand_smiles", "canonical_smiles")
INCHIKEY_COLUMNS = (
    "molecule_inchikey",
    "inchikey",
    "ligand_inchikey",
    "standard_inchi_key",
    "standard_inchikey",
)
ACTIVITY_ID_COLUMNS = ("activity_identity", "activity_id", "evidence_id")
ACTIVITY_TYPE_COLUMNS = (
    "activity_type",
    "affinity_type",
    "standard_type",
    "act_type",
)
ACTIVITY_VALUE_COLUMNS = (
    "activity_value",
    "affinity_value",
    "standard_value",
    "act_value",
    "pchembl",
    "pchembl_value",
)
ACTIVITY_RELATION_COLUMNS = ("relation", "standard_relation")
ACTIVITY_UNIT_COLUMNS = ("activity_unit", "affinity_unit", "standard_units", "act_units")
EXACT_DATE_COLUMNS = (
    "evidence_date",
    "publication_date",
    "curation_date",
    "document_date",
    "pub_date",
)
YEAR_DATE_COLUMNS = ("document_year", "publication_year", "year")


def _clean(value: object) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.upper() in {"", "NA", "N/A", "NULL", "NAN", "NONE"} else text


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_column(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    by_lower = {column.strip().lower(): column for column in columns}
    for alias in aliases:
        found = by_lower.get(alias.lower())
        if found is not None:
            return found
    return None


def _first_present(row: pd.Series, columns: Iterable[str]) -> str:
    for column in columns:
        value = _clean(row.get(column))
        if value:
            return value
    return ""


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"Evidence input required and must be non-empty: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix == ".csv":
        return pd.read_csv(path)
    raise SystemExit(f"Unsupported evidence input format for {path}; expected parquet, tsv, txt, or csv")


def _parse_date_value(value: object) -> tuple[str, str] | None:
    text = _clean(value)
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d-%b-%Y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat(), "exact"
        except ValueError:
            pass
    if len(text) >= 10:
        try:
            return date.fromisoformat(text[:10]).isoformat(), "exact"
        except ValueError:
            pass
    if text.isdigit() and len(text) == 4:
        year = int(text)
        if 1000 <= year <= 2999:
            return f"{year}-12-31", "year_conservative_dec31"
    return None


def _parse_year_value(value: object) -> tuple[str, str] | None:
    text = _clean(value)
    if not text:
        return None
    try:
        year = int(float(text))
    except ValueError:
        return None
    if 1000 <= year <= 2999:
        return f"{year}-12-31", "year_conservative_dec31"
    return None


def _evidence_date(row: pd.Series, exact_cols: list[str], year_cols: list[str]) -> tuple[str, str, str]:
    for column in exact_cols:
        parsed = _parse_date_value(row.get(column))
        if parsed is not None and parsed[1] == "exact":
            return parsed[0], column, parsed[1]
    for column in exact_cols:
        parsed = _parse_date_value(row.get(column))
        if parsed is not None:
            return parsed[0], column, parsed[1]
    for column in year_cols:
        parsed = _parse_year_value(row.get(column))
        if parsed is not None:
            return parsed[0], column, parsed[1]
    return "", "", "undated"


def _activity_identity(row: pd.Series, columns: dict[str, str | None], label: str) -> str:
    direct = _first_present(row, [col for col in [columns["activity_id"]] if col])
    if direct:
        return direct
    parts = [
        label,
        _first_present(row, [col for col in [columns["activity_type"]] if col]),
        _first_present(row, [col for col in [columns["activity_relation"]] if col]),
        _first_present(row, [col for col in [columns["activity_value"]] if col]),
        _first_present(row, [col for col in [columns["activity_unit"]] if col]),
    ]
    if not any(parts[1:]):
        return ""
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _validate_columns(df: pd.DataFrame, path: Path) -> dict[str, str | None]:
    columns = list(df.columns)
    mapped: dict[str, str | None] = {
        "source_db": _find_column(columns, SOURCE_DB_COLUMNS),
        "source_release": _find_column(columns, SOURCE_RELEASE_COLUMNS),
        "source_license": _find_column(columns, SOURCE_LICENSE_COLUMNS),
        "target_uniprot": _find_column(columns, TARGET_COLUMNS),
        "molecule_id": _find_column(columns, MOLECULE_ID_COLUMNS),
        "molecule_smiles": _find_column(columns, SMILES_COLUMNS),
        "molecule_inchikey": _find_column(columns, INCHIKEY_COLUMNS),
        "activity_id": _find_column(columns, ACTIVITY_ID_COLUMNS),
        "activity_type": _find_column(columns, ACTIVITY_TYPE_COLUMNS),
        "activity_relation": _find_column(columns, ACTIVITY_RELATION_COLUMNS),
        "activity_value": _find_column(columns, ACTIVITY_VALUE_COLUMNS),
        "activity_unit": _find_column(columns, ACTIVITY_UNIT_COLUMNS),
    }
    exact_dates = [col for col in (_find_column(columns, (alias,)) for alias in EXACT_DATE_COLUMNS) if col]
    year_dates = [col for col in (_find_column(columns, (alias,)) for alias in YEAR_DATE_COLUMNS) if col]
    missing: list[str] = []
    for key in ("source_db", "source_release", "source_license", "target_uniprot"):
        if mapped[key] is None:
            missing.append(key)
    if not any(mapped[key] for key in ("molecule_id", "molecule_smiles", "molecule_inchikey")):
        missing.append("molecule identity")
    if mapped["activity_id"] is None and (
        mapped["activity_type"] is None or mapped["activity_value"] is None
    ):
        missing.append("activity identity")
    if not exact_dates and not year_dates:
        missing.append("usable date field")
    if missing:
        raise SystemExit(f"{path} missing required normalized evidence column(s): {', '.join(missing)}")
    mapped["exact_date_columns"] = "|".join(exact_dates)
    mapped["year_date_columns"] = "|".join(year_dates)
    return mapped


def _fail_blank_required(df: pd.DataFrame, path: Path, columns: dict[str, str | None]) -> None:
    errors: list[str] = []
    required = {
        "source_db": columns["source_db"],
        "source_release": columns["source_release"],
        "source_license": columns["source_license"],
        "target_uniprot": columns["target_uniprot"],
    }
    for name, column in required.items():
        assert column is not None
        blanks = df[column].map(_clean).eq("")
        if bool(blanks.any()):
            errors.append(f"{name} blank rows={blanks[blanks].index.tolist()[:5]}")
    molecule_cols = [
        col
        for col in (
            columns["molecule_id"],
            columns["molecule_smiles"],
            columns["molecule_inchikey"],
        )
        if col
    ]
    molecule_blank = df.apply(lambda row: not _first_present(row, molecule_cols), axis=1)
    if bool(molecule_blank.any()):
        errors.append(f"molecule identity blank rows={molecule_blank[molecule_blank].index.tolist()[:5]}")
    if columns["activity_id"] is None:
        activity_cols = [columns["activity_type"], columns["activity_value"]]
    else:
        activity_cols = [columns["activity_id"]]
    activity_blank = df.apply(
        lambda row: not _first_present(row, [col for col in activity_cols if col]),
        axis=1,
    )
    if bool(activity_blank.any()):
        errors.append(f"activity identity blank rows={activity_blank[activity_blank].index.tolist()[:5]}")
    if errors:
        raise SystemExit(f"{path} has incomplete required evidence identity: " + "; ".join(errors))


def _normalize_one(path: Path, label: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = _read_table(path)
    columns = _validate_columns(raw, path)
    _fail_blank_required(raw, path, columns)
    exact_cols = str(columns["exact_date_columns"]).split("|") if columns["exact_date_columns"] else []
    year_cols = str(columns["year_date_columns"]).split("|") if columns["year_date_columns"] else []
    normalized = raw.copy()
    normalized.insert(0, "input_label", label)
    normalized.insert(1, "input_path", str(path))
    if "input_row_number" in normalized.columns:
        source_rows = pd.to_numeric(
            normalized.pop("input_row_number"),
            errors="coerce",
        )
        if (
            source_rows.isna().any()
            or source_rows.lt(1).any()
            or source_rows.mod(1).ne(0).any()
        ):
            raise SystemExit(
                f"{path} has invalid input_row_number values; expected positive integers"
            )
        normalized.insert(2, "input_row_number", source_rows.astype("int64"))
    else:
        normalized.insert(2, "input_row_number", range(1, len(normalized) + 1))
    normalized["source_db_norm"] = raw[columns["source_db"]].map(_clean)  # type: ignore[index]
    normalized["source_release_norm"] = raw[columns["source_release"]].map(_clean)  # type: ignore[index]
    normalized["source_license_norm"] = raw[columns["source_license"]].map(_clean)  # type: ignore[index]
    normalized["target_uniprot"] = raw[columns["target_uniprot"]].map(lambda value: _clean(value).upper())  # type: ignore[index]
    normalized["molecule_id_norm"] = (
        raw[columns["molecule_id"]].map(_clean) if columns["molecule_id"] else ""
    )
    normalized["molecule_smiles_norm"] = (
        raw[columns["molecule_smiles"]].map(_clean) if columns["molecule_smiles"] else ""
    )
    normalized["molecule_inchikey_norm"] = (
        raw[columns["molecule_inchikey"]].map(lambda value: _clean(value).upper())
        if columns["molecule_inchikey"]
        else ""
    )
    normalized["activity_identity"] = raw.apply(
        lambda row: _activity_identity(row, columns, label),
        axis=1,
    )
    date_parts = raw.apply(lambda row: _evidence_date(row, exact_cols, year_cols), axis=1)
    normalized["evidence_date"] = [part[0] for part in date_parts]
    normalized["evidence_date_source"] = [part[1] for part in date_parts]
    normalized["evidence_date_precision"] = [part[2] for part in date_parts]
    meta = {
        "path": str(path),
        "label": label,
        "sha256": _sha256_file(path),
        "rows": int(len(raw)),
        "columns": list(raw.columns),
        "mapped_columns": columns,
    }
    return normalized, meta


@functools.lru_cache(maxsize=65536)
def _standardized_parent_inchikey(smiles: str) -> str | None:
    """Standardized parent InChIKey for raw SMILES, or None if unresolved."""
    try:
        from build_activity_retrieval_index import _standardize_mol
        from rdkit import Chem
    except ImportError:
        return None
    try:
        mol = _standardize_mol(str(smiles).strip())
        key = str(Chem.MolToInchiKey(mol) or "").strip().upper()
    except Exception:  # noqa: BLE001 - RDKit parse/standardize errors are broad
        return None
    return key if len(key) == 27 else None


def _structure_identity(row: dict[str, object]) -> tuple[str, str]:
    """Split identity for one evidence row.

    The primary key is the standardized parent InChIKey derived from SMILES, so
    a source that reports a salt and a source that reports the parent land in
    the same split group and a source that supplies only an InChIKey still
    matches. Source molecule IDs and InChIKeys are aliases, never the key when
    a structure can be resolved. Rows with neither are kept under an explicit
    unresolved alias namespace and flagged in `molecule_key_source`.
    """
    smiles = _clean(row.get("molecule_smiles_norm"))
    if smiles:
        key = _standardized_parent_inchikey(smiles)
        if key:
            return f"inchikey:{key}", "standardized_parent_inchikey"
    inchikey = _clean(row.get("molecule_inchikey_norm")).upper()
    if inchikey:
        return f"inchikey:{inchikey}", "source_inchikey"
    molecule_id = _clean(row.get("molecule_id_norm"))
    if molecule_id:
        source_db = _clean(row.get("source_db_norm"))
        return f"unresolved:{source_db}:{molecule_id}", "unresolved_source_molecule_id"
    if smiles:
        return f"unresolved_smiles:{smiles}", "unresolved_raw_smiles"
    return "unresolved:blank", "unresolved_blank"


def _add_split_and_cold_flags(df: pd.DataFrame, cutoff: date) -> pd.DataFrame:
    out = df.copy()
    dated = out["evidence_date"].map(_clean).ne("")
    parsed_dates = pd.to_datetime(out.loc[dated, "evidence_date"], errors="raise").dt.date
    out["temporal_split"] = "undated"
    out.loc[parsed_dates[parsed_dates <= cutoff].index, "temporal_split"] = "pre_cutoff"
    out.loc[parsed_dates[parsed_dates > cutoff].index, "temporal_split"] = "post_cutoff"
    identity_columns = [
        "molecule_smiles_norm",
        "molecule_inchikey_norm",
        "molecule_id_norm",
        "source_db_norm",
    ]
    identities = [
        _structure_identity(record)
        for record in out[identity_columns].to_dict("records")
    ]
    out["molecule_key"] = [identity[0] for identity in identities]
    out["molecule_key_source"] = [identity[1] for identity in identities]
    out["target_key"] = out["target_uniprot"].map(_clean)
    out["pair_key"] = out["molecule_key"] + "|" + out["target_key"]
    pre = out["temporal_split"].eq("pre_cutoff")
    pre_pairs = set(out.loc[pre, "pair_key"])
    pre_molecules = set(out.loc[pre, "molecule_key"])
    pre_targets = set(out.loc[pre, "target_key"])
    out["pair_seen_pre_cutoff"] = out["pair_key"].isin(pre_pairs)
    out["molecule_seen_pre_cutoff"] = out["molecule_key"].isin(pre_molecules)
    out["target_seen_pre_cutoff"] = out["target_key"].isin(pre_targets)
    evidence_payloads = out[
        [
            "input_label",
            "input_path",
            "input_row_number",
            "source_db_norm",
            "source_release_norm",
            "target_uniprot",
            "molecule_key",
            "activity_identity",
            "evidence_date",
        ]
    ].astype(str)
    out["evidence_id"] = [
        hashlib.sha256(json.dumps(record, sort_keys=True).encode("utf-8")).hexdigest()
        for record in evidence_payloads.to_dict("records")
    ]
    return out.sort_values(
        ["evidence_date", "input_label", "input_path", "input_row_number", "evidence_id"],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)


def _tmp_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def _remove_outputs(out_dir: Path) -> None:
    for filename in list(OUTPUT_FILES.values()) + [MANIFEST_NAME]:
        path = out_dir / filename
        path.unlink(missing_ok=True)
        _tmp_path(path).unlink(missing_ok=True)
    staging = out_dir / ".evidence_splits.staging"
    if staging.exists():
        shutil.rmtree(staging)


def _write_outputs(frames: dict[str, pd.DataFrame], manifest: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    staging = out_dir / ".evidence_splits.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        staged_paths: dict[str, Path] = {}
        for key, filename in OUTPUT_FILES.items():
            path = staging / filename
            frames[key].to_parquet(path, index=False)
            staged_paths[key] = path
        manifest["output_sha256"] = {
            filename: _sha256_file(staged_paths[key])
            for key, filename in OUTPUT_FILES.items()
        }
        manifest["outputs"] = {
            filename: {
                "path": str(out_dir / filename),
                "bytes": staged_paths[key].stat().st_size,
                "sha256": manifest["output_sha256"][filename],
                "rows": int(len(frames[key])),
            }
            for key, filename in OUTPUT_FILES.items()
        }
        manifest["manifest_sha256_policy"] = (
            "split_manifest.json is excluded from output_sha256 because self-hashing is not stable"
        )
        manifest_stage = staging / MANIFEST_NAME
        manifest_stage.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest["outputs"][MANIFEST_NAME] = {
            "path": str(out_dir / MANIFEST_NAME),
            "bytes": 0,
            "rows": 1,
        }
        while True:
            rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            rendered_bytes = len(rendered.encode("utf-8"))
            if manifest["outputs"][MANIFEST_NAME]["bytes"] == rendered_bytes:
                manifest_stage.write_text(rendered, encoding="utf-8")
                break
            manifest["outputs"][MANIFEST_NAME]["bytes"] = rendered_bytes

        for key, filename in OUTPUT_FILES.items():
            staged_paths[key].replace(out_dir / filename)
        manifest_stage.replace(out_dir / MANIFEST_NAME)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def parse_evidence_specs(values: list[str]) -> list[tuple[Path, str]]:
    specs: list[tuple[Path, str]] = []
    labels: set[str] = set()
    for value in values:
        if "=" not in value:
            raise SystemExit("--evidence inputs must be path=label")
        path_text, label = value.rsplit("=", 1)
        if not path_text or not label:
            raise SystemExit("--evidence inputs must be path=label")
        if label in labels:
            raise SystemExit(f"Duplicate --evidence label is not allowed: {label}")
        labels.add(label)
        specs.append((Path(path_text), label))
    return specs


def build_splits(evidence_specs: list[tuple[Path, str]], cutoff_date: str, out_dir: Path) -> dict[str, Any]:
    cutoff = date.fromisoformat(cutoff_date)
    out_dir.mkdir(parents=True, exist_ok=True)
    _remove_outputs(out_dir)
    normalized_frames: list[pd.DataFrame] = []
    inputs: list[dict[str, Any]] = []
    try:
        for path, label in evidence_specs:
            frame, meta = _normalize_one(path, label)
            normalized_frames.append(frame)
            inputs.append(meta)
        all_rows = pd.concat(normalized_frames, ignore_index=True) if normalized_frames else pd.DataFrame()
        if all_rows.empty:
            raise SystemExit("No evidence rows found in inputs")
        split_rows = _add_split_and_cold_flags(all_rows, cutoff)
        frames = {
            "pre_cutoff": split_rows[split_rows["temporal_split"].eq("pre_cutoff")].copy(),
            "post_cutoff": split_rows[split_rows["temporal_split"].eq("post_cutoff")].copy(),
            "undated": split_rows[split_rows["temporal_split"].eq("undated")].copy(),
            "post_cutoff_novel_pairs": split_rows[
                split_rows["temporal_split"].eq("post_cutoff")
                & ~split_rows["pair_seen_pre_cutoff"]
            ].copy(),
        }
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "cutoff_date": cutoff_date,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": inputs,
            "input_sha256": {item["label"]: item["sha256"] for item in inputs},
            "schema": {
                "normalized_columns": [
                    "evidence_id",
                    "source_db_norm",
                    "source_release_norm",
                    "source_license_norm",
                    "target_uniprot",
                    "molecule_id_norm",
                    "molecule_smiles_norm",
                    "molecule_inchikey_norm",
                    "molecule_key",
                    "molecule_key_source",
                    "activity_identity",
                    "evidence_date",
                    "evidence_date_source",
                    "evidence_date_precision",
                    "temporal_split",
                    "pair_seen_pre_cutoff",
                    "molecule_seen_pre_cutoff",
                    "target_seen_pre_cutoff",
                ],
                "version": SCHEMA_VERSION,
            },
            "policy": {
                "cutoff_boundary": "evidence_date <= cutoff_date is pre_cutoff",
                "date_preference": (
                    "exact evidence/publication/curation/document dates in alias order; "
                    "year-only dates are converted to Dec 31 conservatively"
                ),
                "fail_closed": (
                    "input schema must include target, molecule identity, source db/release/license, "
                    "activity identity, and at least one usable date field; required identities must be nonblank"
                ),
                "cold_flags": "pair/molecule/target seen in pre_cutoff evidence only",
                "novel_pairs": "post_cutoff_novel_pairs excludes molecule-target pairs seen pre_cutoff",
                "negative_policy": "no negatives are created",
                "row_policy": "all evidence rows and assay/activity types are preserved",
                "molecule_identity": {
                    "primary_key": (
                        "standardized parent InChIKey from SMILES (fragment parent + "
                        "uncharge, shared with the retrieval index standardizer)"
                    ),
                    "aliases": [
                        "normalized source InChIKey when the row has no SMILES",
                        "source molecule ID only for rows with no structure identity",
                    ],
                    "source_inchikey_limitation": (
                        "an InChIKey is a hash and cannot be defragmented; a source "
                        "that supplies only a salt InChIKey without SMILES remains a "
                        "distinct structure rather than being guessed into a parent"
                    ),
                    "unresolved_policy": (
                        "rows without a parseable SMILES or a source InChIKey keep an "
                        "explicit unresolved alias key; they are flagged by "
                        "molecule_key_source and never merged into resolved structures"
                    ),
                },
            },
            "counts": {
                "all_rows": int(len(split_rows)),
                "pre_cutoff": int(len(frames["pre_cutoff"])),
                "post_cutoff": int(len(frames["post_cutoff"])),
                "undated": int(len(frames["undated"])),
                "post_cutoff_novel_pairs": int(len(frames["post_cutoff_novel_pairs"])),
                "unresolved_molecule_identities": int(
                    (
                        ~split_rows["molecule_key_source"].isin(
                            {"standardized_parent_inchikey", "source_inchikey"}
                        )
                    ).sum()
                ),
            },
        }
        return _write_outputs(frames, manifest, out_dir)
    except BaseException:
        _remove_outputs(out_dir)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", action="append", required=True, help="Input as path=label")
    parser.add_argument("--cutoff-date", required=True, help="YYYY-MM-DD cutoff; boundary is pre_cutoff")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        date.fromisoformat(args.cutoff_date)
    except ValueError as exc:
        raise SystemExit("--cutoff-date must be YYYY-MM-DD") from exc
    args.evidence_specs = parse_evidence_specs(args.evidence)
    return args


def main() -> None:
    args = parse_args()
    manifest = build_splits(args.evidence_specs, args.cutoff_date, args.out_dir)
    print(
        "wrote evidence splits all={all_rows} pre={pre_cutoff} post={post_cutoff} "
        "undated={undated} novel_pairs={post_cutoff_novel_pairs}".format(**manifest["counts"])
    )


if __name__ == "__main__":
    main()
