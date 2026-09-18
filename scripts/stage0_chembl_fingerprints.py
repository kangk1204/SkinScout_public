#!/usr/bin/env python3
"""stage0_chembl_fingerprints.py

Pre-compute Morgan (ECFP4 ~ radius 2, 2048-bit) fingerprints for every unique
SMILES in `human_activities.parquet`. Write to
`<chembl_dir>/fp_morgan2_2048.parquet` with columns:
    molecule_chembl_id, smiles, bitvec (32 decimal uint64 words)

Downstream similarity searches reload this once at startup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

LOG = logging.getLogger("stage0.chembl.fp")
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
FINGERPRINT_SCHEMA_VERSION = "chembl_fingerprint_snapshot.v1"


def _remove_outputs(*paths: Path) -> None:
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _parquet_metadata(path: Path, required_columns: set[str]) -> tuple[int, list[str]]:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"Required parquet is missing or empty: {path}")
    try:
        parquet = pq.ParquetFile(path)
        columns = list(parquet.schema_arrow.names)
        rows = int(parquet.metadata.num_rows)
    except Exception as exc:
        raise SystemExit(f"Unable to inspect parquet metadata: {path}") from exc
    missing = sorted(required_columns - set(columns))
    if missing:
        raise SystemExit(f"Parquet missing required columns {missing}: {path}")
    if rows < 1:
        raise SystemExit(f"Parquet contains no rows: {path}")
    return rows, columns


def write_fingerprint_manifest(
    *,
    input_path: Path,
    output_path: Path,
    source_manifest_path: Path,
    manifest_path: Path,
) -> dict[str, object]:
    input_rows, input_columns = _parquet_metadata(
        input_path,
        {"molecule_chembl_id", "smiles"},
    )
    output_rows, output_columns = _parquet_metadata(
        output_path,
        {"molecule_chembl_id", "smiles", "bitvec"},
    )
    if output_rows > input_rows:
        raise SystemExit(
            "Fingerprint row count cannot exceed source activity rows: "
            f"fingerprints={output_rows} activities={input_rows}"
        )
    if not source_manifest_path.exists() or source_manifest_path.stat().st_size == 0:
        raise SystemExit(f"ChEMBL source manifest is required: {source_manifest_path}")
    try:
        source_manifest = json.loads(source_manifest_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise SystemExit(
            f"Unable to parse ChEMBL source manifest: {source_manifest_path}"
        ) from exc
    source = source_manifest.get("source")
    if not isinstance(source, dict):
        raise SystemExit(
            f"ChEMBL source manifest missing source object: {source_manifest_path}"
        )
    release = str(source.get("release", "")).strip()
    license_name = str(source.get("license", "")).strip()
    if not release or not license_name:
        raise SystemExit(
            "ChEMBL source manifest requires nonblank source.release and "
            f"source.license: {source_manifest_path}"
        )
    payload: dict[str, object] = {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": {
            "name": "Morgan ECFP4",
            "radius": 2,
            "n_bits": 2048,
            "use_chirality": False,
            "storage": "32 little-endian uint64 decimal words; MSB-first bits per byte",
        },
        "source_snapshot": {
            "manifest_path": str(source_manifest_path),
            "manifest_sha256": _sha256(source_manifest_path),
            "source_name": str(source.get("name", "ChEMBL")),
            "source_release": release,
            "source_license": license_name,
        },
        "input": {
            "path": str(input_path),
            "sha256": _sha256(input_path),
            "rows": input_rows,
            "columns": input_columns,
        },
        "artifact": {
            "path": str(output_path),
            "sha256": _sha256(output_path),
            "bytes": output_path.stat().st_size,
            "rows": output_rows,
            "columns": output_columns,
        },
    }
    _write_json_atomic(payload, manifest_path)
    return payload


def bits_to_uint64(bv: "DataStructs.ExplicitBitVect") -> np.ndarray:
    arr = np.zeros(2048, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    # Persist an explicit little-endian word format so the parquet is portable
    # across host byte orders.  The bit order within each byte remains NumPy's
    # default big-endian order for compatibility with existing snapshots.
    return np.packbits(arr).view("<u8").copy()


def smiles_to_fp(smi: str) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    bv = MORGAN_GENERATOR.GetFingerprint(mol)
    return bits_to_uint64(bv)


def uint64_words_for_parquet(row: np.ndarray) -> list[str]:
    return [str(int(word)) for word in row]


def _canonical_smiles_key(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return text
    return Chem.MolToSmiles(mol, canonical=True)


def reject_conflicting_molecule_smiles(df: pd.DataFrame, path: Path) -> None:
    normalized = pd.DataFrame({
        "molecule_chembl_id": df["molecule_chembl_id"].fillna("").astype(str).str.strip(),
        "smiles": df["smiles"].map(_canonical_smiles_key),
    })
    blank_ids = normalized.index[normalized["molecule_chembl_id"] == ""].tolist()
    if blank_ids:
        shown = ",".join(str(value) for value in blank_ids[:10])
        suffix = "..." if len(blank_ids) > 10 else ""
        raise SystemExit(
            "ChEMBL human_activities.parquet contains blank "
            f"molecule_chembl_id values at row index {shown}{suffix}: {path}"
        )
    counts = normalized.groupby("molecule_chembl_id", dropna=False)["smiles"].nunique()
    conflicts = [idx for idx, count in counts.items() if count > 1]
    if conflicts:
        shown = ",".join(str(value) for value in conflicts[:10])
        suffix = "..." if len(conflicts) > 10 else ""
        raise SystemExit(
            "ChEMBL human_activities.parquet contains duplicate "
            "molecule_chembl_id values with conflicting SMILES: "
            f"{shown}{suffix}: {path}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chembl-dir", required=True, type=Path)
    parser.add_argument("--out-name", default="fp_morgan2_2048.parquet")
    parser.add_argument("--manifest-name", default="fingerprint_manifest.json")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="validate an existing fingerprint parquet and write only its manifest",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50000,
        help="log progress after every N molecules; use 0 to disable progress logs",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    in_parquet = args.chembl_dir / "human_activities.parquet"
    out_path = args.chembl_dir / args.out_name
    source_manifest_path = args.chembl_dir / "source_manifest.json"
    manifest_path = args.chembl_dir / args.manifest_name
    if args.reuse_existing:
        write_fingerprint_manifest(
            input_path=in_parquet,
            output_path=out_path,
            source_manifest_path=source_manifest_path,
            manifest_path=manifest_path,
        )
        LOG.info("Wrote %s", manifest_path)
        return
    _remove_outputs(out_path, manifest_path)
    if not in_parquet.exists():
        raise SystemExit(f"Missing {in_parquet}; run mirror first.")

    df = pd.read_parquet(in_parquet, columns=["molecule_chembl_id", "smiles"])
    reject_conflicting_molecule_smiles(df, in_parquet)
    df = df.drop_duplicates("molecule_chembl_id").reset_index(drop=True)
    LOG.info("Unique molecules: %d", len(df))

    fps: list[np.ndarray] = []
    for idx, row in df.iterrows():
        smi = row["smiles"]
        if smi is None or not str(smi).strip():
            raise SystemExit(
                "ChEMBL human_activities.parquet contains blank SMILES at "
                f"row index {idx}: {row['molecule_chembl_id']}"
            )
        fp = smiles_to_fp(str(smi).strip())
        if fp is None:
            raise SystemExit(
                "ChEMBL human_activities.parquet contains invalid SMILES at "
                f"row index {idx}: {row['molecule_chembl_id']} -> {smi!r}"
            )
        fps.append(fp)
        done = idx + 1
        if args.progress_every > 0 and (
            done == len(df) or done % args.progress_every == 0
        ):
            LOG.info("Computed fingerprints: %d/%d", done, len(df))
    if not fps:
        raise SystemExit(
            f"No valid ChEMBL SMILES fingerprints could be computed from {in_parquet}"
        )
    arr = np.stack(fps)
    LOG.info("Computed fingerprints: %d", arr.shape[0])

    out_df = pd.DataFrame({
        "molecule_chembl_id": df["molecule_chembl_id"],
        "smiles": df["smiles"],
        "bitvec": [uint64_words_for_parquet(row) for row in arr],
    })
    _write_parquet_atomic(out_df, out_path)
    try:
        write_fingerprint_manifest(
            input_path=in_parquet,
            output_path=out_path,
            source_manifest_path=source_manifest_path,
            manifest_path=manifest_path,
        )
    except BaseException:
        _remove_outputs(out_path, manifest_path)
        raise
    LOG.info("Wrote %s", out_path)
    LOG.info("Wrote %s", manifest_path)


if __name__ == "__main__":
    main()
