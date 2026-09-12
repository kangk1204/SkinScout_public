#!/usr/bin/env python3
"""Build a provenance-bound train-only activity retrieval index."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize


BENCHMARK_SCHEMA_VERSION = "activity_benchmark.v1"
SCHEMA_VERSION = "skinscout.activity-retrieval-index.v4"
# A production index may read dev and test as well. It gets its own schema and
# its own role marker so the evaluation gate cannot consume it by accident - the
# temporal split is the whole basis of every recovery number we report.
PRODUCTION_SCHEMA_VERSION = "skinscout.activity-retrieval-index.production.v1"
EVALUATION_ROLE = "evaluation"
PRODUCTION_ROLE = "production"
OPTIONAL_SPLITS = ("dev", "test")
REQUIRED_COLUMNS = {
    "split",
    "source_db",
    "uniprot",
    "ligand_smiles",
    "publication_key",
    "endpoint",
    "pactivity",
}
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
    includeChirality=False,
)
ENDPOINT_FAMILIES = {
    "KI": "direct_binding",
    "KD": "direct_binding",
    "IC50": "functional",
    "EC50": "functional",
}
LOG = logging.getLogger("activity-retrieval-index")


def _remove_outputs(*paths: Path) -> None:
    for path in paths:
        path.unlink(missing_ok=True)
        path.with_suffix(path.suffix + ".tmp").unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"Benchmark manifest is missing or empty: {path}")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Unable to parse benchmark manifest: {path}") from exc
    if payload.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        raise SystemExit(
            "Benchmark manifest schema_version must be "
            f"{BENCHMARK_SCHEMA_VERSION}: {path}"
        )
    return payload


def _expected_split_sha(manifest: dict[str, Any], split: str) -> str:
    output_sha256 = manifest.get("output_sha256")
    if not isinstance(output_sha256, dict):
        raise SystemExit("Benchmark manifest missing output_sha256 object")
    value = output_sha256.get(f"{split}.parquet")
    if not isinstance(value, str) or len(value) != 64:
        raise SystemExit(f"Benchmark manifest missing output_sha256['{split}.parquet']")
    return value


def _expected_split_rows(manifest: dict[str, Any], split: str) -> int:
    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        raise SystemExit("Benchmark manifest missing splits object")
    counts = splits.get("counts")
    if not isinstance(counts, dict) or split not in counts:
        raise SystemExit(f"Benchmark manifest missing splits.counts.{split}")
    try:
        rows = int(counts[split])
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Benchmark manifest has invalid splits.counts.{split}") from exc
    if rows < 1:
        raise SystemExit(f"Benchmark manifest splits.counts.{split} must be positive")
    return rows


def _expected_train_sha(manifest: dict[str, Any]) -> str:
    return _expected_split_sha(manifest, "train")


def _expected_train_rows(manifest: dict[str, Any]) -> int:
    return _expected_split_rows(manifest, "train")


def _inspect_split_parquet(path: Path, split: str) -> tuple[int, list[str]]:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"{split.capitalize()} parquet is missing or empty: {path}")
    try:
        parquet = pq.ParquetFile(path)
    except Exception as exc:
        raise SystemExit(f"Unable to inspect {split} parquet: {path}") from exc
    columns = list(parquet.schema_arrow.names)
    missing = sorted(REQUIRED_COLUMNS - set(columns))
    if missing:
        raise SystemExit(f"{split.capitalize()} parquet missing required columns {missing}: {path}")
    rows = int(parquet.metadata.num_rows)
    if rows < 1:
        raise SystemExit(f"{split.capitalize()} parquet contains no rows: {path}")
    return rows, columns


def _inspect_train_parquet(path: Path) -> tuple[int, list[str]]:
    return _inspect_split_parquet(path, "train")


def _validate_split_provenance(
    parquet_path: Path, manifest: dict[str, Any], split: str
) -> tuple[str, int]:
    actual_sha = _sha256(parquet_path)
    expected_sha = _expected_split_sha(manifest, split)
    if actual_sha != expected_sha:
        raise SystemExit(
            f"{split.capitalize()} parquet sha256 does not match benchmark manifest "
            f"output_sha256['{split}.parquet']: {actual_sha} != {expected_sha}"
        )
    actual_rows, _ = _inspect_split_parquet(parquet_path, split)
    expected_rows = _expected_split_rows(manifest, split)
    if actual_rows != expected_rows:
        raise SystemExit(
            f"{split.capitalize()} parquet row count does not match benchmark manifest "
            f"splits.counts.{split}: {actual_rows} != {expected_rows}"
        )
    return actual_sha, actual_rows


def _validate_provenance(train_parquet: Path, manifest: dict[str, Any]) -> tuple[str, int]:
    return _validate_split_provenance(train_parquet, manifest, "train")


def _nonblank(value: object) -> bool:
    return not pd.isna(value) and str(value).strip() != ""


def _validate_train_rows(
    df: pd.DataFrame, path: Path, allowed_splits: tuple[str, ...] = ("train",)
) -> None:
    for column in sorted(REQUIRED_COLUMNS):
        if column not in df.columns:
            raise SystemExit(f"Train parquet missing required column {column!r}: {path}")
    permitted = set(allowed_splits)
    bad_split = df.index[~df["split"].astype(str).isin(permitted)].tolist()
    if bad_split:
        raise SystemExit(
            "Activity retrieval index requires every input row to have split in "
            f"{sorted(permitted)}; bad row index {bad_split[:10]}"
        )
    for column in ["source_db", "uniprot", "ligand_smiles", "publication_key", "endpoint"]:
        invalid = [int(idx) for idx, value in df[column].items() if not _nonblank(value)]
        if invalid:
            raise SystemExit(
                f"Train parquet contains blank {column} values at row index "
                f"{invalid[:10]}: {path}"
            )
    pactivity = pd.to_numeric(df["pactivity"], errors="coerce")
    invalid_pactivity = [
        int(idx)
        for idx, value in pactivity.items()
        if pd.isna(value) or not math.isfinite(float(value))
    ]
    if invalid_pactivity:
        raise SystemExit(
            "Train parquet contains non-finite pactivity values at row index "
            f"{invalid_pactivity[:10]}: {path}"
        )
    df["pactivity"] = pactivity.astype(float)
    endpoints = df["endpoint"].astype(str).str.strip().str.upper()
    unsupported = sorted(set(endpoints) - set(ENDPOINT_FAMILIES))
    if unsupported:
        raise SystemExit(
            "Train parquet contains unsupported endpoint values "
            f"{unsupported[:10]}; expected {sorted(ENDPOINT_FAMILIES)}: {path}"
        )
    df["endpoint"] = endpoints


def _parse_smiles(raw_smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(raw_smiles)
    if mol is None:
        params = Chem.SmilesParserParams()
        params.strictCXSMILES = False
        mol = Chem.MolFromSmiles(raw_smiles, params)
    if mol is None:
        raise ValueError(f"invalid ligand_smiles: {raw_smiles!r}")
    return mol


def _canonical_reparse(mol: Chem.Mol) -> Chem.Mol | None:
    """Canonical SMILES round-trip, with a kekulé retry for aromatic edge cases."""
    for kekule in (False, True):
        try:
            canonical = Chem.MolToSmiles(
                mol,
                canonical=True,
                isomericSmiles=True,
                kekuleSmiles=kekule,
            )
        except (RuntimeError, ValueError):
            continue
        reparsed = Chem.MolFromSmiles(canonical)
        if reparsed is not None:
            return reparsed
    return None


def standardize_parent(mol: Chem.Mol) -> Chem.Mol:
    """Fragment parent, neutralised, shared by the index and the run-path query.

    Uncharger's output does not always parse back (anionic phosphinates raise on
    re-parse), and if the index drops such a record while a run keeps it, the
    two sides score different structures for the same compound. Falling back to
    the charged fragment parent keeps the record with a valid structure instead
    of failing the build. Audit H-11 asked for one canonicalisation function so
    index and query cannot drift; this is it.
    """
    parent = rdMolStandardize.FragmentParent(mol)
    uncharged = _canonical_reparse(rdMolStandardize.Uncharger().uncharge(parent))
    if uncharged is not None:
        return uncharged
    charged = _canonical_reparse(parent)
    if charged is None:
        raise ValueError("standardized parent is invalid")
    return charged


def _standardize_mol(raw_smiles: str) -> Chem.Mol:
    mol = _parse_smiles(raw_smiles)
    try:
        return standardize_parent(mol)
    except ValueError as exc:
        raise ValueError(f"standardized ligand_smiles is invalid: {raw_smiles!r}") from exc


def _standardize_ligand(raw_smiles: str) -> tuple[str, str, str, list[str]]:
    parent = _standardize_mol(raw_smiles)
    canonical = Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True)
    ligand_key = Chem.MolToInchiKey(parent)
    if not ligand_key:
        raise ValueError(f"unable to derive InChIKey for ligand_smiles: {raw_smiles!r}")
    fingerprint = MORGAN_GENERATOR.GetFingerprint(parent)
    return ligand_key, ligand_key[:14], canonical, _uint64_words_for_parquet(fingerprint)


def _uint64_words_for_parquet(bv: DataStructs.ExplicitBitVect) -> list[str]:
    arr = np.zeros(2048, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return [str(int(word)) for word in np.packbits(arr).view("<u8").copy()]


def _structure_ligand_key(
    standard_inchikey: str,
    canonical_smiles: str,
) -> tuple[str, str]:
    if not standard_inchikey or not canonical_smiles:
        raise ValueError("standard_inchikey and canonical_smiles must be nonblank")
    suffix = hashlib.sha256(canonical_smiles.encode("utf-8")).hexdigest()[:12]
    return (
        f"{standard_inchikey}#SMILES-{suffix}",
        "fragment_parent_canonical_smiles_key",
    )


def _standardize_ligand_record(
    raw_smiles: str,
) -> tuple[str, tuple[str, str] | None, dict[str, object]]:
    """Standardize one ligand, reporting failure rather than raising.

    A ProcessPoolExecutor.map raises on the first bad item, which makes one
    malformed record able to stop a 1.4M-row rebuild. The caller decides whether
    an unusable ligand is fatal - see --max-unusable-ligands, which is 0 by
    default so the evaluation index keeps failing closed.
    """
    try:
        standard_inchikey, connectivity_key, canonical, bitvec = _standardize_ligand(
            raw_smiles
        )
    except ValueError as exc:
        return raw_smiles, None, {"error": str(exc)}
    identity = (standard_inchikey, canonical)
    return (
        raw_smiles,
        identity,
        {
            "standard_inchikey": standard_inchikey,
            "connectivity_key": connectivity_key,
            "canonical_smiles": canonical,
            "bitvec": bitvec,
        },
    )


def _build_ligands(
    df: pd.DataFrame,
    workers: int = 1,
    progress_every: int = 50_000,
    max_unusable: int = 0,
) -> tuple[pd.DataFrame, dict[str, str], list[dict[str, str]]]:
    by_identity: dict[tuple[str, str], dict[str, object]] = {}
    raw_smiles_to_identity: dict[str, tuple[str, str]] = {}
    unusable: list[dict[str, str]] = []
    raw_smiles_values = sorted({str(value).strip() for value in df["ligand_smiles"]})
    if workers < 1:
        raise SystemExit("workers must be positive")
    LOG.info(
        "Standardizing %d unique ligands with %d worker(s)",
        len(raw_smiles_values),
        workers,
    )
    with ExitStack() as stack:
        if workers == 1:
            iterator = map(_standardize_ligand_record, raw_smiles_values)
        else:
            pool = stack.enter_context(ProcessPoolExecutor(max_workers=workers))
            iterator = pool.map(
                _standardize_ligand_record,
                raw_smiles_values,
                chunksize=256,
            )
        try:
            for done, (raw_smiles, identity, record) in enumerate(iterator, start=1):
                if identity is None:
                    unusable.append(
                        {"ligand_smiles": raw_smiles, "error": str(record.get("error", ""))}
                    )
                    continue
                raw_smiles_to_identity[raw_smiles] = identity
                existing = by_identity.get(identity)
                if existing is None:
                    by_identity[identity] = record
                elif existing["bitvec"] != record["bitvec"]:
                    raise SystemExit(
                        "Conflicting fingerprints for identical standardized canonical "
                        f"SMILES: {identity[0]} {identity[1]}"
                    )
                if progress_every > 0 and (
                    done == len(raw_smiles_values) or done % progress_every == 0
                ):
                    LOG.info(
                        "Standardized ligands: %d/%d",
                        done,
                        len(raw_smiles_values),
                    )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    if len(unusable) > max_unusable:
        listed = "; ".join(item["error"] for item in unusable[:5])
        raise SystemExit(
            f"{len(unusable)} ligand(s) could not be standardized, over the allowance "
            f"of {max_unusable}. Raise --max-unusable-ligands only after looking at "
            f"them (scripts/audit_unusable_ligands.py). First: {listed}"
        )
    if unusable:
        LOG.warning(
            "dropped %d unusable ligand(s) within the allowance of %d; "
            "they are listed in the index manifest",
            len(unusable),
            max_unusable,
        )

    identity_to_ligand_key: dict[tuple[str, str], str] = {}
    records: list[dict[str, object]] = []
    for identity, record in sorted(by_identity.items()):
        standard_inchikey, canonical = identity
        ligand_key, route = _structure_ligand_key(
            standard_inchikey,
            canonical,
        )
        identity_to_ligand_key[identity] = ligand_key
        records.append(
            {
                "ligand_key": ligand_key,
                **record,
                "standardization_route": route,
            }
        )
    rows = [
        {"ligand_index": idx, **record}
        for idx, record in enumerate(sorted(records, key=lambda row: str(row["ligand_key"])))
    ]
    raw_smiles_to_key = {
        raw_smiles: identity_to_ligand_key[identity]
        for raw_smiles, identity in raw_smiles_to_identity.items()
    }
    ligands = pd.DataFrame(
        rows,
        columns=[
            "ligand_index",
            "ligand_key",
            "standard_inchikey",
            "connectivity_key",
            "canonical_smiles",
            "standardization_route",
            "bitvec",
        ],
    ).astype({"ligand_index": "int64"})
    return ligands, raw_smiles_to_key, unusable


def _build_edges(
    df: pd.DataFrame,
    ligands: pd.DataFrame,
    raw_smiles_to_key: dict[str, str],
    negative_threshold: float,
    positive_threshold: float,
) -> pd.DataFrame:
    key_to_index = dict(zip(ligands["ligand_key"], ligands["ligand_index"], strict=True))
    # A ligand dropped within the allowance has no key, so its rows cannot become
    # edges. Selecting them out here keeps the lookup total instead of raising.
    usable = df["ligand_smiles"].astype(str).str.strip().isin(raw_smiles_to_key)
    if not usable.all():
        LOG.warning("dropping %d row(s) whose ligand could not be standardized", int((~usable).sum()))
        df = df.loc[usable].reset_index(drop=True)
    work = pd.DataFrame(
        {
            "ligand_index": [
                int(key_to_index[raw_smiles_to_key[str(value).strip()]])
                for value in df["ligand_smiles"]
            ],
            "uniprot": df["uniprot"].astype(str).str.strip(),
            "source_db": df["source_db"].astype(str).str.strip(),
            "pactivity": df["pactivity"].astype(float),
            "endpoint": df["endpoint"].astype(str).str.strip(),
            "endpoint_family": [
                ENDPOINT_FAMILIES[str(value).strip().upper()]
                for value in df["endpoint"]
            ],
            "publication_key": df["publication_key"].astype(str).str.strip(),
        }
    )
    work["positive"] = work["pactivity"] >= positive_threshold
    work["negative"] = work["pactivity"] <= negative_threshold
    work["gray"] = ~(work["positive"] | work["negative"])
    group_columns = ["ligand_index", "uniprot", "source_db", "endpoint_family"]
    edges = (
        work.groupby(group_columns, sort=True, as_index=False)
        .agg(
            measurement_count=("pactivity", "size"),
            positive_measurement_count=("positive", "sum"),
            gray_measurement_count=("gray", "sum"),
            negative_measurement_count=("negative", "sum"),
            max_pactivity=("pactivity", "max"),
            median_pactivity=("pactivity", "median"),
            endpoint_count=("endpoint", "nunique"),
            publication_count=("publication_key", "nunique"),
        )
        .sort_values(group_columns)
        .reset_index(drop=True)
    )
    count_columns = [
        "measurement_count",
        "positive_measurement_count",
        "gray_measurement_count",
        "negative_measurement_count",
        "endpoint_count",
        "publication_count",
    ]
    edges[count_columns] = edges[count_columns].astype("int64")
    return edges


def _write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _manifest_payload(
    *,
    train_parquet: Path,
    benchmark_manifest: Path,
    out_ligands: Path,
    out_edges: Path,
    train_sha256: str,
    train_rows: int,
    ligands: pd.DataFrame,
    edges: pd.DataFrame,
    negative_threshold: float,
    positive_threshold: float,
    source_counts: Counter[str],
    index_role: str = EVALUATION_ROLE,
    extra_splits: dict[str, dict[str, Any]] | None = None,
    unusable_ligands: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    extra_splits = dict(extra_splits or {})
    unusable_ligands = list(unusable_ligands or [])
    splits: dict[str, dict[str, Any]] = {
        "train": {
            "path": str(train_parquet.resolve()),
            "sha256": train_sha256,
            "rows": train_rows,
        }
    }
    splits.update(extra_splits)
    return {
        "schema_version": (
            PRODUCTION_SCHEMA_VERSION if index_role == PRODUCTION_ROLE else SCHEMA_VERSION
        ),
        # An evaluation index is train-only so the temporal split holds. A
        # production index also reads dev and test, which makes every recovery
        # number measured against it optimistic. The role travels with the file.
        "index_role": index_role,
        # Every ligand the standardizer could not use, named. An empty list is
        # the normal case; a non-empty one is a deliberate, bounded exclusion.
        "unusable_ligands": {
            "count": len(unusable_ligands),
            "records": unusable_ligands,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "train_parquet": {
                "path": str(train_parquet.resolve()),
                "sha256": train_sha256,
                "rows": train_rows,
            },
            "splits": splits,
            "benchmark_manifest": {
                "path": str(benchmark_manifest.resolve()),
                "sha256": _sha256(benchmark_manifest),
            },
        },
        "outputs": {
            "ligands": {
                "path": str(out_ligands.resolve()),
                "sha256": _sha256(out_ligands),
                "rows": int(len(ligands)),
            },
            "edges": {
                "path": str(out_edges.resolve()),
                "sha256": _sha256(out_edges),
                "rows": int(len(edges)),
            },
        },
        "algorithm": {
            "standardization": [
                "RDKit strict SMILES parse",
                "RDKit strictCXSMILES=False fallback",
                "rdMolStandardize.FragmentParent",
                "rdMolStandardize.Uncharger",
                "canonical isomeric SMILES",
                "deterministic Kekule SMILES fallback only when aromatic serialization cannot be sanitized",
                "standard InChIKey and 14-character connectivity block",
                "dataset-independent ligand key: standard InChIKey plus SHA256 prefix of canonical SMILES",
                "distinct canonical structures sharing a standard InChIKey remain explicit structure keys",
            ],
            "fingerprint": {
                "name": "Morgan ECFP4",
                "radius": 2,
                "n_bits": 2048,
                "use_chirality": False,
                "storage": "32 little-endian uint64 decimal words; MSB-first bits per byte",
            },
            "label_policy": {
                "positive": f"measured pActivity >= {positive_threshold:g}",
                "gray": (
                    f"measured {negative_threshold:g} < pActivity < "
                    f"{positive_threshold:g}; excluded from binary calibration labels"
                ),
                "negative": f"measured pActivity <= {negative_threshold:g}",
                "unmeasured_pairs": "unlabeled; never fabricated as negatives",
                "negative_threshold": negative_threshold,
                "positive_threshold": positive_threshold,
            },
            "endpoint_families": ENDPOINT_FAMILIES,
            "aggregation": [
                "ligand_index",
                "uniprot",
                "source_db",
                "endpoint_family",
            ],
            "structure_key_count": int(len(ligands)),
            "standard_inchikey_collision_groups": int(
                (ligands.groupby("standard_inchikey", sort=False).size() > 1).sum()
            ),
            "standard_inchikey_collision_structures": int(
                ligands["standard_inchikey"].duplicated(keep=False).sum()
            ),
        },
        "versions": {
            "rdkit": rdkit.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "source_counts": dict(sorted(source_counts.items())),
    }


def _extra_split_paths(args: argparse.Namespace) -> dict[str, Path]:
    """The optional splits a production index reads, in a fixed order."""
    found = {}
    for split in OPTIONAL_SPLITS:
        path = getattr(args, f"{split}_parquet", None)
        if path is not None:
            found[split] = path
    return found


def build_index(args: argparse.Namespace) -> None:
    out_paths = [args.out_ligands, args.out_edges, args.out_manifest]
    index_role = getattr(args, "index_role", EVALUATION_ROLE)
    extras = _extra_split_paths(args)
    try:
        manifest = _load_manifest(args.benchmark_manifest)
        train_sha256, train_rows = _validate_provenance(args.train_parquet, manifest)
        df = pd.read_parquet(args.train_parquet)
        _validate_train_rows(df, args.train_parquet)
        if len(df) != train_rows:
            raise SystemExit(
                "Loaded train parquet row count changed after metadata validation: "
                f"{len(df)} != {train_rows}"
            )
        extra_manifest: dict[str, dict[str, Any]] = {}
        for split, path in extras.items():
            split_sha, split_rows = _validate_split_provenance(path, manifest, split)
            frame = pd.read_parquet(path)
            _validate_train_rows(frame, path, allowed_splits=(split,))
            if len(frame) != split_rows:
                raise SystemExit(
                    f"Loaded {split} parquet row count changed after metadata "
                    f"validation: {len(frame)} != {split_rows}"
                )
            extra_manifest[split] = {
                "path": str(path.resolve()),
                "sha256": split_sha,
                "rows": split_rows,
            }
            df = pd.concat([df, frame], ignore_index=True)
            LOG.info("added %s split: %d rows (total %d)", split, len(frame), len(df))
        source_counts = Counter(df["source_db"].astype(str).str.strip())
        ligands, raw_smiles_to_key, unusable = _build_ligands(
            df,
            workers=args.workers,
            progress_every=args.progress_every,
            max_unusable=getattr(args, "max_unusable_ligands", 0),
        )
        edges = _build_edges(
            df,
            ligands,
            raw_smiles_to_key,
            args.negative_threshold,
            args.positive_threshold,
        )
        _write_parquet_atomic(ligands, args.out_ligands)
        _write_parquet_atomic(edges, args.out_edges)
        payload = _manifest_payload(
            train_parquet=args.train_parquet,
            benchmark_manifest=args.benchmark_manifest,
            out_ligands=args.out_ligands,
            out_edges=args.out_edges,
            train_sha256=train_sha256,
            train_rows=train_rows,
            ligands=ligands,
            edges=edges,
            negative_threshold=args.negative_threshold,
            positive_threshold=args.positive_threshold,
            source_counts=source_counts,
            index_role=index_role,
            extra_splits=extra_manifest,
            unusable_ligands=unusable,
        )
        _write_json_atomic(payload, args.out_manifest)
    except BaseException:
        _remove_outputs(*out_paths)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-parquet", required=True, type=Path)
    parser.add_argument("--benchmark-manifest", required=True, type=Path)
    parser.add_argument(
        "--index-role",
        choices=(EVALUATION_ROLE, PRODUCTION_ROLE),
        default=EVALUATION_ROLE,
        help=(
            "evaluation (default) reads train only, so the temporal split holds; "
            "production also reads --dev-parquet/--test-parquet and must never be "
            "used to measure recovery"
        ),
    )
    parser.add_argument(
        "--dev-parquet",
        type=Path,
        help="dev split; requires --index-role production",
    )
    parser.add_argument(
        "--test-parquet",
        type=Path,
        help="test split; requires --index-role production",
    )
    parser.add_argument("--out-ligands", required=True, type=Path)
    parser.add_argument("--out-edges", required=True, type=Path)
    parser.add_argument("--out-manifest", required=True, type=Path)
    parser.add_argument("--negative-threshold", type=float, default=5.0)
    parser.add_argument("--positive-threshold", type=float, default=6.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel ligand standardization workers (default: 1)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50_000,
        help="log progress after every N unique ligands; use 0 to disable",
    )
    parser.add_argument(
        "--max-unusable-ligands",
        type=int,
        default=0,
        help=(
            "how many ligands may fail standardization before the build stops "
            "(default 0). Every dropped ligand is named in the index manifest; "
            "look at them with scripts/audit_unusable_ligands.py before raising this"
        ),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not math.isfinite(args.negative_threshold):
        raise SystemExit("--negative-threshold must be finite")
    if not math.isfinite(args.positive_threshold):
        raise SystemExit("--positive-threshold must be finite")
    if args.negative_threshold >= args.positive_threshold:
        raise SystemExit("--negative-threshold must be less than --positive-threshold")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.progress_every < 0:
        raise SystemExit("--progress-every must be non-negative")
    if args.max_unusable_ligands < 0:
        raise SystemExit("--max-unusable-ligands must be non-negative")
    extras = _extra_split_paths(args)
    if args.index_role == EVALUATION_ROLE and extras:
        raise SystemExit(
            "An evaluation index is train-only; "
            f"pass --index-role production to read {sorted(extras)}"
        )
    if args.index_role == PRODUCTION_ROLE and not extras:
        raise SystemExit(
            "--index-role production needs at least one of --dev-parquet/--test-parquet; "
            "otherwise it is the evaluation index under a misleading name"
        )
    build_index(args)


if __name__ == "__main__":
    main()
