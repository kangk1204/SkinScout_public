#!/usr/bin/env python3
"""stage0_verify.py

Run §3.4 verification checklist over the data/ tree.
Exits 0 only if every check passes unless --allow-soft-failures is explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from Bio.SeqUtils.CheckSum import crc64

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from data_readiness import (  # noqa: E402
    apply_config_overrides,
    default_stage0_paths,
    load_workflow_config,
    resolve_stage0_paths,
)
from stage0_manifest import (  # noqa: E402
    failed_preview as _pdbqt_failed_preview,
    load_manifest as _load_pdbqt_manifest,
    manifest_path as _pdbqt_manifest_path,
    validate_manifest as _validate_pdbqt_manifest,
)

LOG = logging.getLogger("stage0.verify")


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def chk_alphafold_count(root: Path, expected: int = 20300) -> Check:
    pdbs = list(root.rglob("AF-*-model_v*.pdb"))
    return Check("alphafold_count", len(pdbs) >= expected,
                 f"found {len(pdbs)} AF PDBs (expected ≥ {expected})")


def chk_cleaned_count(clean_dir: Path, expected: int = 20000) -> Check:
    pdbs = list(clean_dir.glob("*_clean.pdb"))
    empty = [p for p in pdbs if p.stat().st_size < 200]
    ok = len(pdbs) >= expected and not empty
    detail = f"{len(pdbs)} cleaned PDBs; {len(empty)} suspiciously empty"
    return Check("cleaned_count_and_size", ok, detail)


def _preview(items: list[str], limit: int = 5) -> str:
    if not items:
        return "none"
    suffix = "..." if len(items) > limit else ""
    return ", ".join(items[:limit]) + suffix


def _clean_target_ids(clean_dir: Path) -> set[str]:
    return {
        pdb.stem.removesuffix("_clean")
        for pdb in clean_dir.glob("*_clean.pdb")
    }


def _collect_paths(
    repo: Path,
    overrides: Mapping[str, Path] | None = None,
) -> dict[str, Path]:
    """Resolve the shared Stage 0 path map, overrides winning over defaults."""
    resolved = default_stage0_paths(repo)
    for key, value in (overrides or {}).items():
        if key not in resolved:
            raise ValueError(f"unknown Stage 0 path key: {key}")
        resolved[key] = Path(value)
    return resolved


def _load_target_list(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    try:
        lines = path.read_text().splitlines()
    except (OSError, UnicodeError):
        return set()
    return {
        line.strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    }


def chk_target_list_file(path: Path) -> Check:
    if not path.is_file():
        return Check(
            "claim_no_pocket_targets_list",
            False,
            f"{path} missing or not a regular file; empty file is valid",
        )
    try:
        path.read_text()
    except (OSError, UnicodeError) as exc:
        return Check(
            "claim_no_pocket_targets_list",
            False,
            f"{path} is unreadable: {exc}",
        )
    return Check(
        "claim_no_pocket_targets_list",
        True,
        f"{path} present; empty file is valid",
    )


def _pocket_manifest_payloads(pocket_dir: Path) -> tuple[dict[str, dict], list[str]]:
    payloads: dict[str, dict] = {}
    bad: list[str] = []
    for path in sorted(pocket_dir.glob("*.pockets.json")):
        target = path.name.removesuffix(".pockets.json")
        try:
            doc = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            bad.append(target)
            continue
        if not isinstance(doc, dict):
            bad.append(target)
            continue
        payloads[target] = doc
    return payloads, bad


def _with_pocket_targets(pocket_dir: Path) -> set[str]:
    payloads, _bad = _pocket_manifest_payloads(pocket_dir)
    return {
        target
        for target, doc in payloads.items()
        if isinstance(doc.get("pockets"), list) and len(doc["pockets"]) > 0
    }


def _finite_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _valid_pocket_record(record: object) -> bool:
    if not isinstance(record, dict):
        return False
    center = record.get("center")
    rank = record.get("rank")
    radius = record.get("radius")
    score = record.get("score")
    druggability = record.get("druggability")
    return (
        isinstance(rank, int)
        and not isinstance(rank, bool)
        and rank > 0
        and _finite_number(score)
        and float(score) >= 0.0
        and _finite_number(druggability)
        and isinstance(center, list)
        and len(center) == 3
        and all(_finite_number(value) for value in center)
        and _finite_number(radius)
        and float(radius) > 0.0
    )


def chk_p2rank(
    pocket_dir: Path,
    expected_targets: set[str] | None = None,
    no_pocket_targets: set[str] | None = None,
) -> Check:
    pkts = list(pocket_dir.glob("*.pockets.json"))
    if not pkts:
        return Check("p2rank_no_nan", False, "no pocket manifests found")
    invalid_manifests = 0
    empty_targets: set[str] = set()
    found_targets: set[str] = set()
    for p in sorted(pkts):
        target = p.name.removesuffix(".pockets.json")
        found_targets.add(target)
        try:
            doc = json.loads(p.read_text())
            pockets = doc.get("pockets", [])
            if not isinstance(doc, dict) or doc.get("uniprot") != target:
                invalid_manifests += 1
                continue
            if not isinstance(pockets, list):
                invalid_manifests += 1
                continue
            if not pockets:
                empty_targets.add(target)
            if any(not _valid_pocket_record(pocket) for pocket in pockets):
                invalid_manifests += 1
        except (AttributeError, json.JSONDecodeError, OSError):
            invalid_manifests += 1
    missing: list[str] = []
    unexpected: list[str] = []
    no_pocket_mismatch: list[str] = []
    if expected_targets is not None:
        missing = sorted(expected_targets - found_targets)
        unexpected = sorted(found_targets - expected_targets)
        no_pocket_targets = no_pocket_targets or set()
        no_pocket_mismatch = sorted(empty_targets ^ no_pocket_targets)
    ok = not (invalid_manifests or missing or unexpected or no_pocket_mismatch)
    detail = (
        f"{len(pkts)} pocket manifests, {invalid_manifests} invalid/schema issue; "
        f"missing={len(missing)} [{_preview(missing)}] "
        f"unexpected={len(unexpected)} [{_preview(unexpected)}] "
        f"no_pocket_mismatch={len(no_pocket_mismatch)} "
        f"[{_preview(no_pocket_mismatch)}]"
    )
    return Check("p2rank_no_nan", ok, detail)


def chk_pdbqt_roundtrip(
    pdbqt_dir: Path,
    expected_targets: set[str] | None = None,
    *,
    manifest_path: Path | None = None,
    expected_policy: Mapping[str, Any] | None = None,
) -> Check:
    pdbqts = list(pdbqt_dir.glob("*.pdbqt"))
    if not pdbqts and manifest_path is None:
        return Check("pdbqt_roundtrip", False, "no PDBQT files found")
    # Don't shell out from verifier — just sanity-check header lines.
    bad = 0
    found_targets: set[str] = set()
    for p in sorted(pdbqts):
        found_targets.add(p.stem)
        head = p.read_text(errors="ignore").splitlines()[:5]
        if not any(line.startswith(("ROOT", "ATOM", "HETATM", "REMARK")) for line in head):
            bad += 1
    missing: list[str] = []
    unexpected: list[str] = []
    if expected_targets is not None:
        missing = sorted(expected_targets - found_targets)
        unexpected = sorted(found_targets - expected_targets)

    manifest_errors: list[str] = []
    manifest_failed: dict[str, str] = {}
    if manifest_path is not None:
        payload, load_error = _load_pdbqt_manifest(manifest_path)
        if load_error is not None:
            manifest_errors.append(load_error)
        else:
            index, errors = _validate_pdbqt_manifest(
                payload,
                pdbqt_dir=pdbqt_dir,
                expected_policy=expected_policy,
            )
            manifest_errors.extend(errors)
            if index is not None:
                success = index["success"]
                manifest_failed = index["failed"]
                expected_all = success | set(manifest_failed)
                extras = sorted(found_targets - expected_all)
                if extras:
                    manifest_errors.append(
                        f"unexpected pdbqt files not in manifest: {_preview(extras)}"
                    )
                absent = sorted(success - found_targets)
                if absent:
                    manifest_errors.append(
                        f"manifest success targets without a pdbqt file: {_preview(absent)}"
                    )

    ok = not (bad or missing or unexpected or manifest_errors)
    detail = f"{len(pdbqts)} pdbqt; {bad} suspect headers"
    if manifest_path is None:
        detail += (
            f"; missing={len(missing)} [{_preview(missing)}]"
            f"; unexpected={len(unexpected)} [{_preview(unexpected)}]"
        )
    else:
        # The manifest is the producer's own record of its expected set, so it
        # owns the verdict once validated; the pockets-derived set is a
        # diagnostic here, not an exact-set requirement.
        ok = not (bad or manifest_errors)
        detail += (
            f"; manifest_failed={_pdbqt_failed_preview(manifest_failed)}"
            f"; manifest_errors={_preview(manifest_errors)}"
            f"; pockets_missing={len(missing)} [{_preview(missing)}]"
        )
    return Check("pdbqt_roundtrip", ok, detail)


def chk_mmseqs(mmseqs_dir: Path) -> Check:
    must = ["human_db", "human_db.dbtype", "human_db.lookup"]
    missing = [m for m in must if not (mmseqs_dir / m).exists()]
    return Check("mmseqs_index", not missing, f"missing: {missing or 'none'}")


def chk_mmseqs_training_cutoff(mmseqs_dir: Path) -> Check:
    must = [
        "training_cutoff_seqs.fasta",
        "training_cutoff_db.dbtype",
        "training_cutoff_db.lookup",
    ]
    missing = [m for m in must if not (mmseqs_dir / m).exists()]
    return Check(
        "claim_mmseqs_training_cutoff",
        not missing,
        f"missing: {missing or 'none'}",
    )


def _read_canonical_fasta(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    accession: str | None = None
    chunks: list[str] = []

    def finish_record() -> None:
        if accession is None:
            return
        sequence = "".join(chunks)
        if not sequence:
            raise ValueError(f"FASTA record {accession} has an empty sequence")
        if accession in records:
            raise ValueError(f"FASTA contains duplicate accession {accession}")
        invalid = sorted(set(sequence) - set("ACDEFGHIKLMNPQRSTUVWYO"))
        if invalid:
            raise ValueError(
                f"FASTA record {accession} contains unknown residues {''.join(invalid)}"
            )
        records[accession] = sequence

    with path.open(encoding="ascii") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            if not line:
                raise ValueError(f"FASTA contains a blank line at {line_number}")
            if line.startswith(">"):
                finish_record()
                accession = line[1:]
                chunks = []
                if not accession or any(character.isspace() for character in accession):
                    raise ValueError(f"invalid canonical FASTA header at line {line_number}")
            else:
                if accession is None:
                    raise ValueError("FASTA sequence appears before the first header")
                if line != line.upper() or any(character.isspace() for character in line):
                    raise ValueError(f"invalid canonical FASTA sequence line {line_number}")
                chunks.append(line)
    finish_record()
    if not records:
        raise ValueError("canonical FASTA contains no records")
    return records


def chk_canonical_sequences(
    fasta_path: Path,
    manifest_path: Path,
    clean_dir: Path,
) -> Check:
    """Validate the AFDB-derived canonical FASTA contract without rebuilding sources."""
    name = "claim_canonical_human_sequences"
    errors: list[str] = []

    def reject(detail: str) -> None:
        if len(errors) < 20:
            errors.append(detail)

    if not fasta_path.is_file() or fasta_path.stat().st_size == 0:
        return Check(name, False, f"canonical FASTA missing or empty: {fasta_path}")
    if not manifest_path.is_file() or manifest_path.stat().st_size == 0:
        return Check(name, False, f"canonical manifest missing or empty: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sequences = _read_canonical_fasta(fasta_path)
    except (json.JSONDecodeError, OSError, UnicodeError, ValueError) as exc:
        return Check(name, False, f"canonical FASTA/manifest failed to parse: {exc}")
    if not isinstance(manifest, dict):
        return Check(name, False, "canonical manifest root must be an object")

    if manifest.get("schema_version") != "skinscout.afdb-v4-canonical-sequences.v1":
        reject("schema_version")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        reject("provenance")
        provenance = {}
    if provenance.get("source") != (
        "AlphaFold Protein Structure Database human proteome v4 raw mmCIF files"
    ):
        reject("provenance.source")
    derivation = provenance.get("derivation")
    if not isinstance(derivation, str) or not all(
        phrase in derivation
        for phrase in (
            "assembled offline from AFDB fragment metadata",
            "full UniProt CRC64 embedded by AFDB",
            "not an independently downloaded UniProt FASTA snapshot",
        )
    ):
        reject("provenance.derivation")
    if provenance.get("source_pattern") != "*.cif.gz":
        reject("provenance.source_pattern")

    validation = manifest.get("validation")
    if not isinstance(validation, dict):
        reject("validation")
        validation = {}
    expected_validation = {
        "taxonomy_id": "9606",
        "checksum_algorithm": "Bio.SeqUtils.CheckSum.crc64",
        "fragment_policy": "all fragments; exact spans; equal overlaps; gapless from residue 1",
        "unknown_residues_allowed": False,
    }
    for field, expected in expected_validation.items():
        if validation.get(field) != expected:
            reject(f"validation.{field}")

    artifact = manifest.get("artifact")
    if not isinstance(artifact, dict):
        reject("artifact")
        artifact = {}
    actual_bytes = fasta_path.stat().st_size
    actual_sha256 = _sha256_file(fasta_path)
    if artifact.get("path") != str(fasta_path.resolve()):
        reject("artifact.path")
    if artifact.get("sha256") != actual_sha256:
        reject("artifact.sha256")
    if artifact.get("bytes") != actual_bytes:
        reject("artifact.bytes")
    if artifact.get("sequence_count") != len(sequences):
        reject("artifact.sequence_count")

    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        reject("counts")
        counts = {}
    if counts.get("canonical_sequences") != len(sequences):
        reject("counts.canonical_sequences")

    records = manifest.get("sequences")
    if not isinstance(records, list):
        reject("sequences")
        records = []
    records_by_accession: dict[str, dict] = {}
    source_paths: set[str] = set()
    fragment_count = 0
    for record in records:
        if not isinstance(record, dict):
            reject("sequences.record")
            continue
        accession_value = record.get("accession")
        if not isinstance(accession_value, str) or not accession_value:
            reject("sequences.accession")
            continue
        accession = accession_value
        if accession in records_by_accession:
            reject(f"sequences.duplicate:{accession}")
            continue
        records_by_accession[accession] = record
        sequence = sequences.get(accession)
        if sequence is None:
            reject(f"sequences.unexpected:{accession}")
            continue
        if record.get("length") != len(sequence):
            reject(f"sequences.length:{accession}")
        sequence_sha256 = hashlib.sha256(sequence.encode("ascii")).hexdigest()
        if record.get("sequence_sha256") != sequence_sha256:
            reject(f"sequences.sequence_sha256:{accession}")
        if record.get("uniprot_crc64") != crc64(sequence).removeprefix("CRC-"):
            reject(f"sequences.uniprot_crc64:{accession}")
        fragments = record.get("source_fragments")
        if not isinstance(fragments, list) or not fragments:
            reject(f"sequences.source_fragments:{accession}")
            continue
        intervals: list[tuple[int, int]] = []
        for fragment in fragments:
            fragment_count += 1
            if not isinstance(fragment, dict):
                reject(f"sequences.source_fragment:{accession}")
                continue
            source_path = fragment.get("path")
            source_sha256 = fragment.get("compressed_sha256")
            begin = fragment.get("seq_db_align_begin")
            end = fragment.get("seq_db_align_end")
            if (
                not isinstance(source_path, str)
                or not re.fullmatch(r"AF-.+-F[1-9][0-9]*-model_v4\.cif\.gz", source_path)
                or source_path in source_paths
            ):
                reject(f"sequences.source_path:{accession}")
            else:
                source_paths.add(source_path)
            if not isinstance(source_sha256, str) or not re.fullmatch(
                r"[0-9a-f]{64}", source_sha256
            ):
                reject(f"sequences.compressed_sha256:{accession}")
            if (
                not isinstance(begin, int)
                or isinstance(begin, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
                or begin < 1
                or end < begin
            ):
                reject(f"sequences.fragment_range:{accession}")
            else:
                intervals.append((begin, end))
        intervals.sort()
        if intervals:
            covered_end = intervals[0][1]
            if intervals[0][0] != 1:
                reject(f"sequences.fragment_start:{accession}")
            for begin, end in intervals[1:]:
                if begin > covered_end + 1:
                    reject(f"sequences.fragment_gap:{accession}")
                covered_end = max(covered_end, end)
            if covered_end != len(sequence):
                reject(f"sequences.fragment_end:{accession}")

    if set(records_by_accession) != set(sequences):
        missing_records = sorted(set(sequences) - set(records_by_accession))
        reject(f"sequences.missing_records:{_preview(missing_records)}")
    if counts.get("source_files") != fragment_count:
        reject("counts.source_files")
    if fragment_count != len(source_paths):
        reject("sequences.source_file_uniqueness")

    required = _clean_target_ids(clean_dir)
    missing_required = sorted(required - sequences.keys())
    if missing_required:
        reject(f"required_receptors.missing:{_preview(missing_required)}")
    if counts.get("required_receptors") != len(required):
        reject("counts.required_receptors")
    if counts.get("required_receptors_covered") != len(required) - len(missing_required):
        reject("counts.required_receptors_covered")
    coverage = manifest.get("required_receptor_coverage")
    if not isinstance(coverage, dict):
        reject("required_receptor_coverage")
    else:
        if coverage.get("directory") != str(clean_dir.resolve()):
            reject("required_receptor_coverage.directory")
        if coverage.get("missing_accessions") != missing_required:
            reject("required_receptor_coverage.missing_accessions")

    detail = (
        f"sequences={len(sequences)} fragments={fragment_count} "
        f"required_receptors={len(required)} errors={_preview(errors)}"
    )
    return Check(name, not errors, detail)


def chk_flag(manifests_dir: Path) -> Check:
    flag = manifests_dir / "stage0_complete.flag"
    return Check("stage0_flag", flag.exists(), str(flag))


def chk_nonempty_file(path: Path, name: str) -> Check:
    ok = path.exists() and path.stat().st_size > 0
    detail = f"{path} ({path.stat().st_size if path.exists() else 0} bytes)"
    return Check(name, ok, detail)


def chk_table(path: Path, name: str, required_cols: set[str]) -> Check:
    if not path.exists() or path.stat().st_size == 0:
        return Check(name, False, f"{path} missing or empty")
    try:
        import pandas as pd
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
    except Exception as exc:  # noqa: BLE001
        return Check(name, False, f"{path} failed to parse: {exc}")
    missing = sorted(required_cols - set(df.columns))
    ok = not missing and len(df) > 0
    return Check(name, ok, f"rows={len(df)} missing_cols={missing or 'none'}")


SKIN_SCORE_AXES_SCHEMA = "skinscout.skin-score-axes.v2"
SKIN_SCORE_AXES_SEMANTICS = "relative_within_build_expression_context"
SKIN_SCORE_AXES_NORMALIZATION = "per-axis min-max across proteins in this build"
SKIN_SCORE_DECLARED_WEIGHTS = {
    "hpa_tissue": 0.30,
    "hpa_cell": 0.25,
    "proteome": 0.20,
    "gtex": 0.10,
    "sc": 0.15,
}


def _skin_score_axes_errors(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        return ["root_not_object"]
    errors: list[str] = []
    if payload.get("schema_version") != SKIN_SCORE_AXES_SCHEMA:
        errors.append("schema_version")
    if payload.get("score_semantics") != SKIN_SCORE_AXES_SEMANTICS:
        errors.append("score_semantics")
    if payload.get("normalization") != SKIN_SCORE_AXES_NORMALIZATION:
        errors.append("normalization")

    declared = payload.get("declared_weights")
    declared_ok = isinstance(declared, dict) and set(declared) == set(
        SKIN_SCORE_DECLARED_WEIGHTS
    )
    if declared_ok:
        declared_ok = all(
            _finite_number(declared[axis])
            and math.isclose(
                float(declared[axis]),
                weight,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for axis, weight in SKIN_SCORE_DECLARED_WEIGHTS.items()
        )
    if not declared_ok:
        errors.append("declared_weights")

    axes_absent = payload.get("axes_absent")
    if not isinstance(axes_absent, list) or any(
        not isinstance(axis, str) for axis in axes_absent
    ):
        errors.append("axes_absent")
        absent: set[str] = set()
    else:
        absent = set(axes_absent)
        if len(absent) != len(axes_absent) or not absent <= set(
            SKIN_SCORE_DECLARED_WEIGHTS
        ):
            errors.append("axes_absent")

    effective = payload.get("effective_weights")
    if not isinstance(effective, dict):
        errors.append("effective_weights")
    else:
        present = set(SKIN_SCORE_DECLARED_WEIGHTS) - absent
        if set(effective) != present:
            errors.append("effective_weights.keys")
        elif effective:
            values = list(effective.values())
            if any(
                not _finite_number(value) or float(value) <= 0.0
                for value in values
            ):
                errors.append("effective_weights.values")
            elif not math.isclose(
                sum(float(value) for value in values),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-3,
            ):
                errors.append("effective_weights.sum")

    threshold = payload.get("context_support_threshold")
    if not _finite_number(threshold) or float(threshold) < 0.0:
        errors.append("context_support_threshold")
    return errors


def chk_skin_score_axes(tsv_path: Path, name: str) -> Check:
    """Require the SkinScore axes sidecar and its effective-formula contract.

    The producer always writes skin_score.tsv.axes.json, so a completed Stage 0
    without it cannot explain which axes actually contributed or what weights
    the published score used.
    """
    path = tsv_path.with_suffix(tsv_path.suffix + ".axes.json")
    if not path.exists() or path.stat().st_size == 0:
        return Check(name, False, f"{path} missing or empty")
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return Check(name, False, f"{path} failed JSON validation: {exc}")
    errors = _skin_score_axes_errors(payload)
    return Check(name, not errors, f"errors={errors or 'none'}")


def chk_parquet_schema(
    path: Path,
    name: str,
    required_cols: set[str],
    min_rows: int = 1,
) -> Check:
    """Validate a large parquet from metadata without loading it into memory."""
    if not path.exists() or path.stat().st_size == 0:
        return Check(name, False, f"{path} missing or empty")
    try:
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        # ``ParquetSchema.names`` flattens nested fields and reports the list
        # child (for example ``element``) instead of its top-level column name.
        # Evidence fingerprints use a list-valued ``bitvec`` column, so inspect
        # the Arrow schema contract at the table boundary.
        columns = set(parquet.schema_arrow.names)
        rows = int(parquet.metadata.num_rows)
    except Exception as exc:  # noqa: BLE001
        return Check(name, False, f"{path} failed parquet metadata validation: {exc}")
    missing = sorted(required_cols - columns)
    ok = not missing and rows >= min_rows
    return Check(
        name,
        ok,
        f"rows={rows} min_rows={min_rows} missing_cols={missing or 'none'}",
    )


def chk_evidence_manifest(path: Path, name: str, required_keys: set[str]) -> Check:
    if not path.exists() or path.stat().st_size == 0:
        return Check(name, False, f"{path} missing or empty")
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return Check(name, False, f"{path} failed JSON validation: {exc}")
    if not isinstance(payload, dict):
        return Check(name, False, f"{path} root must be an object")
    missing = sorted(required_keys - set(payload))
    return Check(name, not missing, f"missing_keys={missing or 'none'}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_json_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bound_relative_path(
    manifest: Path,
    value: object,
    label: str,
    *,
    allow_empty: bool = False,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is missing")
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"{label} must be relative")
    bound = manifest.parent / path
    if bound.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {bound}")
    resolved = bound.resolve()
    if (
        not resolved.is_file()
        or (not allow_empty and resolved.stat().st_size <= 0)
    ):
        raise ValueError(f"{label} is missing, empty, or unsafe: {resolved}")
    return resolved


def chk_discovery_alias_integrity(repo: Path) -> Check:
    """Validate the relocatable four-source exact-alias package without rebuilding it."""

    name = "discovery_exact_alias_integrity"
    data = repo / "data"
    pubchem_manifest = data / "pubchem" / "source_manifest.json"
    source_dir = data / "discovery_aliases" / "sources"
    source_manifest = source_dir / "source_manifest.json"
    registry_path = source_dir / "source_registry.json"
    alias_manifest = data / "discovery_aliases" / "manifest.json"
    errors: list[str] = []
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        import yaml

        config = yaml.safe_load((repo / "workflow" / "config.yaml").read_text())
        stage0 = config["stage0"]
        for path, label in (
            (pubchem_manifest, "pubchem manifest"),
            (source_manifest, "alias-source manifest"),
            (registry_path, "alias-source registry"),
            (alias_manifest, "alias-map manifest"),
        ):
            if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
                raise ValueError(f"{label} is missing, empty, or unsafe: {path}")

        pubchem = json.loads(pubchem_manifest.read_text())
        if pubchem.get("schema_version") != "skinscout.pubchem-alias-source.v1":
            errors.append("pubchem.schema_version")
        unsigned_pubchem = dict(pubchem)
        declared_pubchem_binding = unsigned_pubchem.pop("binding_sha256", None)
        if declared_pubchem_binding != _stable_json_hash(unsigned_pubchem):
            errors.append("pubchem.binding_sha256")
        pub_source = pubchem.get("source", {})
        pub_artifact_record = pubchem.get("artifact", {})
        pub_artifact = _bound_relative_path(
            pubchem_manifest,
            pub_artifact_record.get("path"),
            "pubchem.artifact.path",
        )
        expected_pubchem = (data / "pubchem" / "CID-SMILES.gz").resolve()
        if pub_artifact != expected_pubchem:
            errors.append("pubchem.artifact.path_mismatch")
        expected_release = stage0["pubchem_alias_release_date"]
        expected_md5 = stage0["pubchem_alias_smiles_md5"]
        expected_url = stage0["pubchem_alias_smiles_url"]
        if (
            pub_source.get("name") != "PubChem"
            or pub_source.get("release") != expected_release
            or pub_source.get("release_date") != expected_release
            or pub_source.get("redistribution") != "allowed"
        ):
            errors.append("pubchem.source_contract")
        if pub_artifact_record.get("url") != expected_url:
            errors.append("pubchem.url_pin")
        if f"/Monthly/{expected_release}/Extras/" not in expected_url:
            errors.append("pubchem.url_not_monthly_snapshot")
        if pub_artifact_record.get("last_modified_date") != expected_release:
            errors.append("pubchem.last_modified_date")
        if pub_artifact_record.get("bytes") != pub_artifact.stat().st_size:
            errors.append("pubchem.bytes")
        if pub_artifact_record.get("sha256") != _sha256_file(pub_artifact):
            errors.append("pubchem.sha256")
        if (
            pub_artifact_record.get("md5") != expected_md5
            or _md5_file(pub_artifact) != expected_md5
        ):
            errors.append("pubchem.md5")
        if not isinstance(pub_artifact_record.get("rows"), int) or pub_artifact_record["rows"] <= 0:
            errors.append("pubchem.rows")
        pubchem_builder = Path(__file__).resolve().parent / "mirror_pubchem_alias_source.py"
        if pubchem.get("builder_script_sha256") != _sha256_file(pubchem_builder):
            errors.append("pubchem.builder_script_sha256")

        source_doc = json.loads(source_manifest.read_text())
        if source_doc.get("schema_version") != "skinscout.discovery-alias-sources.v1":
            errors.append("sources.schema_version")
        unsigned_sources = dict(source_doc)
        source_binding = unsigned_sources.pop("self_binding_sha256", None)
        if source_binding != _stable_json_hash(unsigned_sources):
            errors.append("sources.self_binding_sha256")
        if source_doc.get("source_registry") != "source_registry.json":
            errors.append("sources.registry_path")
        if source_doc.get("source_registry_sha256") != _sha256_file(registry_path):
            errors.append("sources.registry_sha256")
        source_builder = Path(__file__).resolve().parent / "build_discovery_alias_sources.py"
        if source_doc.get("builder", {}).get("sha256") != _sha256_file(source_builder):
            errors.append("sources.builder_script_sha256")
        source_stats = source_doc.get("source_stats")
        pubchem_stats = source_stats.get("PubChem") if isinstance(source_stats, dict) else None
        if not isinstance(pubchem_stats, dict):
            raise ValueError("alias-source manifest PubChem stats are missing")
        cid_stat_keys = (
            "selected_cids",
            "matched_cids",
            "missing_cids",
            "missing_fraction_ppm",
        )
        if any(
            isinstance(pubchem_stats.get(key), bool)
            or not isinstance(pubchem_stats.get(key), int)
            or pubchem_stats[key] < 0
            for key in cid_stat_keys
        ):
            raise ValueError("alias-source manifest PubChem stats are invalid")
        selected_cids = pubchem_stats["selected_cids"]
        matched_cids = pubchem_stats["matched_cids"]
        missing_cids = pubchem_stats["missing_cids"]
        expected_missing_fraction_ppm = (
            (missing_cids * 1_000_000 + selected_cids - 1) // selected_cids
            if selected_cids > 0
            else -1
        )
        if (
            selected_cids <= 0
            or matched_cids <= 0
            or selected_cids != matched_cids + missing_cids
            or pubchem_stats["missing_fraction_ppm"] != expected_missing_fraction_ppm
        ):
            errors.append("sources.PubChem.cid_stats")
        source_policy = source_doc.get("policy")
        if (
            not isinstance(source_policy, dict)
            or source_policy.get("PubChem_max_missing_fraction_ppm") != 100_000
            or expected_missing_fraction_ppm > 100_000
        ):
            errors.append("sources.PubChem.missing_cid_policy")
        audit_name = "pubchem_missing_cids.txt"
        audit_records = source_doc.get("audit_artifacts")
        if not isinstance(audit_records, dict) or set(audit_records) != {audit_name}:
            raise ValueError("alias-source manifest PubChem audit record is incomplete")
        audit_record = audit_records[audit_name]
        audit_path = source_dir / audit_name
        if (
            not isinstance(audit_record, dict)
            or audit_path.is_symlink()
            or not audit_path.is_file()
        ):
            raise ValueError("alias-source PubChem audit is missing or unsafe")
        if (
            audit_record.get("sha256") != _sha256_file(audit_path)
            or audit_record.get("bytes") != audit_path.stat().st_size
        ):
            errors.append("sources.PubChem.audit_hash_bytes")
        audit_rows = 0
        previous_cid = -1
        with audit_path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                cid = line.rstrip("\n")
                if not re.fullmatch(r"[0-9]+", cid):
                    raise ValueError(
                        f"alias-source PubChem audit line {line_number} is invalid"
                    )
                numeric_cid = int(cid)
                if numeric_cid <= previous_cid:
                    raise ValueError("alias-source PubChem audit is not strictly increasing")
                previous_cid = numeric_cid
                audit_rows += 1
        if audit_record.get("rows") != audit_rows or audit_rows != missing_cids:
            errors.append("sources.PubChem.audit_rows")
        registry = json.loads(registry_path.read_text())
        if registry.get("schema_version") != "discovery_alias_source_registry.v1":
            errors.append("registry.schema_version")
        if registry.get("registry_root") != ".":
            errors.append("registry.root")
        registry_sources = registry.get("sources")
        expected_sources = ("ChEMBL", "BindingDB", "GtoPdb", "PubChem")
        if not isinstance(registry_sources, list) or tuple(
            item.get("canonical_name") for item in registry_sources if isinstance(item, dict)
        ) != expected_sources:
            raise ValueError("registry must contain the four canonical sources in order")
        source_inputs = source_doc.get("inputs")
        if not isinstance(source_inputs, dict) or set(source_inputs) != set(expected_sources):
            raise ValueError("alias-source manifest must bind exactly the four source inputs")

        chembl_release = str(
            next(item for item in registry_sources if item["canonical_name"] == "ChEMBL")[
                "release"
            ]
        )
        upstream_manifest_paths = {
            "ChEMBL": data / "chembl37" / "source_manifest.json",
            "BindingDB": data / "bindingdb" / "bindingdb_source_manifest.json",
            "GtoPdb": data / "gtopdb" / "source_manifest.json",
            "PubChem": pubchem_manifest,
        }
        upstream_artifact_paths = {
            "ChEMBL": (
                data
                / "chembl37"
                / f"chembl_{chembl_release}"
                / f"chembl_{chembl_release}_sqlite"
                / f"chembl_{chembl_release}.db"
            ),
            "BindingDB": data / "bindingdb" / "BindingDB_All.tsv",
            "GtoPdb": data / "gtopdb" / "ligands.csv",
            "PubChem": data / "pubchem" / "CID-SMILES.gz",
        }
        upstream_docs = {
            name: json.loads(path.read_text())
            for name, path in upstream_manifest_paths.items()
        }
        if upstream_docs["ChEMBL"].get("schema_version") != "chembl_activity_evidence.v1":
            errors.append("sources.inputs.ChEMBL.manifest_schema")
        if upstream_docs["BindingDB"].get("schema_version") != 1:
            errors.append("sources.inputs.BindingDB.manifest_schema")
        if upstream_docs["GtoPdb"].get("schema_version") != "skinscout.gtopdb-source.v1":
            errors.append("sources.inputs.GtoPdb.manifest_schema")

        upstream_records = {
            "ChEMBL": upstream_docs["ChEMBL"].get("source", {}),
            "BindingDB": upstream_docs["BindingDB"].get("extracted", {}),
            "GtoPdb": upstream_docs["GtoPdb"].get("artifacts", {}).get(
                "ligands.csv", {}
            ),
            "PubChem": upstream_docs["PubChem"].get("artifact", {}),
        }
        upstream_source_meta = {
            "ChEMBL": upstream_docs["ChEMBL"].get("source", {}),
            "BindingDB": {
                **upstream_docs["BindingDB"].get("source", {}),
                "release": upstream_docs["BindingDB"].get("release"),
                "release_date": upstream_docs["BindingDB"].get("release_date"),
            },
            "GtoPdb": upstream_docs["GtoPdb"].get("source", {}),
            "PubChem": upstream_docs["PubChem"].get("source", {}),
        }
        upstream_hash_keys = {
            "ChEMBL": ("source_db_sha256", "source_db_bytes"),
            "BindingDB": ("sha256", "bytes"),
            "GtoPdb": ("sha256", "bytes"),
            "PubChem": ("sha256", "bytes"),
        }
        registry_by_name = {item["canonical_name"]: item for item in registry_sources}
        for source_name in expected_sources:
            input_record = source_inputs.get(source_name)
            if not isinstance(input_record, dict):
                errors.append(f"sources.inputs.{source_name}.record")
                continue
            input_path = _bound_relative_path(
                source_manifest,
                input_record.get("path"),
                f"sources.inputs.{source_name}.path",
            )
            expected_input_path = upstream_artifact_paths[source_name].resolve()
            if input_path != expected_input_path:
                errors.append(f"sources.inputs.{source_name}.path")
            upstream_record = upstream_records[source_name]
            if not isinstance(upstream_record, dict):
                errors.append(f"sources.inputs.{source_name}.upstream_record")
                continue
            hash_key, bytes_key = upstream_hash_keys[source_name]
            expected_input_hash = upstream_record.get(hash_key)
            expected_input_bytes = upstream_record.get(bytes_key)
            if (
                input_record.get("sha256") != expected_input_hash
                or input_record.get("sha256") != _sha256_file(input_path)
            ):
                errors.append(f"sources.inputs.{source_name}.sha256")
            if (
                input_record.get("bytes") != expected_input_bytes
                or input_path.stat().st_size != expected_input_bytes
            ):
                errors.append(f"sources.inputs.{source_name}.bytes")

            manifest_record = input_record.get("manifest")
            if not isinstance(manifest_record, dict):
                errors.append(f"sources.inputs.{source_name}.manifest")
                continue
            bound_manifest = _bound_relative_path(
                source_manifest,
                manifest_record.get("path"),
                f"sources.inputs.{source_name}.manifest.path",
            )
            expected_manifest = upstream_manifest_paths[source_name].resolve()
            if bound_manifest != expected_manifest:
                errors.append(f"sources.inputs.{source_name}.manifest.path")
            if manifest_record.get("sha256") != _sha256_file(bound_manifest):
                errors.append(f"sources.inputs.{source_name}.manifest.sha256")
            if manifest_record.get("bytes") != bound_manifest.stat().st_size:
                errors.append(f"sources.inputs.{source_name}.manifest.bytes")
            upstream_meta = upstream_source_meta[source_name]
            registry_source = registry_by_name[source_name]
            for field in ("release", "release_date", "license", "license_url"):
                if (
                    manifest_record.get(field) != upstream_meta.get(field)
                    or registry_source.get(field) != upstream_meta.get(field)
                ):
                    errors.append(f"sources.inputs.{source_name}.{field}")
        output_hashes = source_doc.get("output_sha256", {})
        output_bytes = source_doc.get("output_bytes", {})
        output_rows = source_doc.get("output_rows", {})
        expected_artifacts = {
            "ChEMBL": "chembl_aliases.parquet",
            "BindingDB": "bindingdb_aliases.parquet",
            "GtoPdb": "gtopdb_aliases.parquet",
            "PubChem": "pubchem_aliases.parquet",
        }
        for source in registry_sources:
            artifact_record = source.get("artifact", {})
            artifact = _bound_relative_path(
                registry_path,
                artifact_record.get("path"),
                f"registry.{source.get('canonical_name')}.artifact",
            )
            expected_artifact = (
                source_dir / expected_artifacts[source["canonical_name"]]
            ).resolve()
            if artifact != expected_artifact:
                errors.append(f"registry.{source.get('canonical_name')}.path")
            if source.get("redistribution") != "allowed":
                errors.append(f"registry.{source.get('canonical_name')}.redistribution")
            if not str(source.get("license_url", "")).startswith("https://"):
                errors.append(f"registry.{source.get('canonical_name')}.license_url")
            parquet = pq.ParquetFile(artifact)
            rows = int(parquet.metadata.num_rows)
            relative_name = artifact.name
            if tuple(parquet.schema_arrow.names) != (
                "smiles",
                "alias",
                "inchikey",
                "source_record_id",
            ):
                errors.append(f"registry.{relative_name}.schema")
            if any(not pa.types.is_string(field.type) for field in parquet.schema_arrow):
                errors.append(f"registry.{relative_name}.types")
            if rows <= 0 or artifact_record.get("rows") != rows:
                errors.append(f"registry.{relative_name}.rows")
            observed_hash = _sha256_file(artifact)
            if (
                artifact_record.get("sha256") != observed_hash
                or output_hashes.get(relative_name) != observed_hash
            ):
                errors.append(f"registry.{relative_name}.sha256")
            if (
                artifact_record.get("bytes") != artifact.stat().st_size
                or output_bytes.get(relative_name) != artifact.stat().st_size
            ):
                errors.append(f"registry.{relative_name}.bytes")
            if output_rows.get(relative_name) != rows:
                errors.append(f"registry.{relative_name}.manifest_rows")

        alias_doc = json.loads(alias_manifest.read_text())
        if alias_doc.get("schema_version") != "discovery_alias_map.v1":
            errors.append("aliases.schema_version")
        unsigned_alias = dict(alias_doc)
        alias_binding = unsigned_alias.pop("binding_sha256", None)
        if alias_binding != _stable_json_hash(unsigned_alias):
            errors.append("aliases.binding_sha256")
        bound_registry = _bound_relative_path(
            alias_manifest,
            alias_doc.get("source_registry"),
            "aliases.source_registry",
        )
        if (
            bound_registry != registry_path.resolve()
            or alias_doc.get("registry_sha256") != _sha256_file(registry_path)
        ):
            errors.append("aliases.registry_binding")
        bound_source_manifest = _bound_relative_path(
            alias_manifest,
            alias_doc.get("source_manifest"),
            "aliases.source_manifest",
        )
        if (
            bound_source_manifest != source_manifest.resolve()
            or alias_doc.get("source_manifest_sha256")
            != _sha256_file(source_manifest)
        ):
            errors.append("aliases.source_manifest_binding")
        alias_builder = Path(__file__).resolve().parent / "build_discovery_alias_map.py"
        if alias_doc.get("builder_script_sha256") != _sha256_file(alias_builder):
            errors.append("aliases.builder_script_sha256")
        expected_source_hashes = {
            source["canonical_name"]: source["artifact"]["sha256"]
            for source in registry_sources
        }
        if alias_doc.get("source_artifact_sha256") != expected_source_hashes:
            errors.append("aliases.source_artifact_sha256")
        output_paths = alias_doc.get("output_paths", {})
        aliases = _bound_relative_path(
            alias_manifest,
            output_paths.get("aliases_parquet"),
            "aliases.output_paths.aliases_parquet",
        )
        direct = _bound_relative_path(
            alias_manifest,
            output_paths.get("direct_reference"),
            "aliases.output_paths.direct_reference",
        )
        if aliases != (alias_manifest.parent / "aliases.parquet").resolve():
            errors.append("aliases.path")
        if direct != (alias_manifest.parent / "direct_exact_reference.smi").resolve():
            errors.append("direct.path")
        alias_parquet = pq.ParquetFile(aliases)
        expected_alias_schema = pa.schema(
            [(column, pa.string()) for column in (
                "alias", "raw_alias", "discovery_key_sha256",
                "parent_canonical_smiles", "parent_inchikey",
                "parent_connectivity_inchikey", "input_smiles", "source",
                "release", "source_record_id", "source_inchikey",
            )]
        )
        if alias_parquet.schema_arrow != expected_alias_schema:
            errors.append("aliases.parquet_schema")
        alias_rows = int(alias_parquet.metadata.num_rows)
        counts = alias_doc.get("counts", {})
        if alias_rows <= 0 or counts.get("alias_rows") != alias_rows:
            errors.append("aliases.rows")
        source_counts = counts.get("source_counts")
        expected_source_rows = {
            source["canonical_name"]: source["artifact"]["rows"]
            for source in registry_sources
        }
        expected_audit_by_source: dict[str, int] = {}
        if not isinstance(source_counts, dict) or set(source_counts) != set(
            expected_sources
        ):
            errors.append("aliases.source_counts")
            source_counts = {}
        for source_name in expected_sources:
            source_count = source_counts.get(source_name)
            if not isinstance(source_count, dict):
                errors.append(f"aliases.source_counts.{source_name}")
                continue
            fields = {
                field: source_count.get(field)
                for field in (
                    "input_rows",
                    "canonicalized_rows",
                    "canonicalization_exclusions",
                    "canonicalization_exclusion_fraction_ppm",
                )
            }
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in fields.values()
            ):
                errors.append(f"aliases.source_counts.{source_name}.values")
                continue
            input_rows = fields["input_rows"]
            exclusions = fields["canonicalization_exclusions"]
            expected_fraction = (
                exclusions * 1_000_000 + input_rows - 1
            ) // input_rows if input_rows > 0 else -1
            if (
                input_rows != expected_source_rows[source_name]
                or fields["canonicalized_rows"] + exclusions != input_rows
                or fields["canonicalization_exclusion_fraction_ppm"]
                != expected_fraction
                or expected_fraction > 10_000
            ):
                errors.append(f"aliases.source_counts.{source_name}.contract")
            expected_audit_by_source[source_name] = exclusions
        outputs = alias_doc.get("outputs", {})
        output_sizes = alias_doc.get("output_bytes", {})
        if (
            outputs.get("aliases_parquet_sha256") != _sha256_file(aliases)
            or output_sizes.get("aliases_parquet") != aliases.stat().st_size
        ):
            errors.append("aliases.output_binding")
        if (
            outputs.get("direct_reference_sha256") != _sha256_file(direct)
            or output_sizes.get("direct_reference") != direct.stat().st_size
        ):
            errors.append("direct.output_binding")
        policy = alias_doc.get("policy")
        if policy != {
            "fail_closed": True,
            "max_canonicalization_exclusion_fraction_ppm_per_source": 10_000,
        }:
            errors.append("aliases.canonicalization_policy")
        audit_record = alias_doc.get("canonicalization_audit")
        if not isinstance(audit_record, dict) or set(audit_record) != {
            "path",
            "sha256",
            "bytes",
            "rows",
        }:
            raise ValueError("canonicalization audit record is missing")
        audit_path = _bound_relative_path(
            alias_manifest,
            audit_record.get("path"),
            "aliases.canonicalization_audit.path",
            allow_empty=True,
        )
        expected_audit_path = (
            alias_manifest.parent / "canonicalization_exclusions.jsonl"
        ).resolve()
        if audit_path != expected_audit_path:
            errors.append("aliases.canonicalization_audit.path")
        if (
            audit_record.get("sha256") != _sha256_file(audit_path)
            or audit_record.get("bytes") != audit_path.stat().st_size
        ):
            errors.append("aliases.canonicalization_audit.output_binding")
        audit_rows = 0
        observed_audit_by_source = {
            source_name: 0 for source_name in expected_sources
        }
        previous_audit: tuple[str, int] | None = None
        with audit_path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.endswith("\n"):
                    raise ValueError(
                        f"canonicalization audit row {line_number} lacks newline"
                    )
                record = json.loads(line)
                if (
                    not isinstance(record, dict)
                    or set(record) != {
                        "input_smiles_sha256",
                        "reason_code",
                        "row_index",
                        "source",
                        "source_record_id",
                    }
                    or record.get("reason_code") != "rdkit_canonicalization_failed"
                    or not re.fullmatch(
                        r"[0-9a-f]{64}",
                        str(record.get("input_smiles_sha256", "")),
                    )
                    or record.get("source") not in expected_sources
                    or isinstance(record.get("row_index"), bool)
                    or not isinstance(record.get("row_index"), int)
                    or record["row_index"] < 0
                    or not isinstance(record.get("source_record_id"), str)
                ):
                    raise ValueError(
                        f"canonicalization audit row {line_number} is invalid"
                    )
                marker = (record["source"], record["row_index"])
                if previous_audit is not None and marker <= previous_audit:
                    raise ValueError(
                        "canonicalization audit rows are not strictly sorted"
                    )
                previous_audit = marker
                canonical_line = json.dumps(
                    record,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ) + "\n"
                if line != canonical_line:
                    raise ValueError(
                        "canonicalization audit is not canonically encoded"
                    )
                observed_audit_by_source[record["source"]] += 1
                audit_rows += 1
        expected_audit_rows = sum(expected_audit_by_source.values())
        if (
            audit_record.get("rows") != audit_rows
            or audit_rows != expected_audit_rows
        ):
            errors.append("aliases.canonicalization_audit.rows")
        for source_name in expected_sources:
            if observed_audit_by_source[source_name] != expected_audit_by_source.get(
                source_name, -1
            ):
                errors.append(
                    f"aliases.canonicalization_audit.{source_name}.rows"
                )
        canonical_payload = {
            "schema_version": alias_doc.get("schema_version"),
            "registry_sha256": alias_doc.get("registry_sha256"),
            "source_manifest_sha256": alias_doc.get("source_manifest_sha256"),
            "source_artifact_sha256": alias_doc.get("source_artifact_sha256"),
            "outputs": outputs,
            "counts": counts,
            "canonicalization_audit": audit_record,
            "policy": policy,
            "canonical_contract": alias_doc.get("canonical_contract"),
            "builder_script_sha256": alias_doc.get("builder_script_sha256"),
        }
        if alias_doc.get("canonical_payload_sha256") != _stable_json_hash(canonical_payload):
            errors.append("aliases.canonical_payload_sha256")
        direct_rows = 0
        previous: tuple[str, str] | None = None
        with direct.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.endswith("\n"):
                    raise ValueError(f"direct reference row {line_number} lacks newline")
                fields = line.rstrip("\n").split(" ")
                if (
                    len(fields) != 2
                    or not fields[0]
                    or not re.fullmatch(r"[0-9a-f]{64}", fields[1])
                ):
                    raise ValueError(f"direct reference row {line_number} is invalid")
                if hashlib.sha256(fields[0].encode("utf-8")).hexdigest() != fields[1]:
                    raise ValueError(
                        f"direct reference row {line_number} key binding is invalid"
                    )
                marker = (fields[0], fields[1])
                if previous is not None and marker <= previous:
                    raise ValueError("direct reference rows are not strictly sorted")
                previous = marker
                direct_rows += 1
        if direct_rows <= 0 or counts.get("direct_reference_rows") != direct_rows:
            errors.append("direct.rows")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"exception:{exc}")
    return Check(name, not errors, f"errors={errors or 'none'}")


def _parquet_row_count(path: Path) -> int:
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(path).metadata.num_rows)


def chk_bindingdb_evidence_integrity(
    bindingdb_dir: Path,
    *,
    expected_release: str = "2026-08",
    expected_database_license: str = "CC BY 3.0",
    expected_evidence_license: str = "CC BY 3.0 (BindingDB-curated)",
    expected_license_url: str = "https://www.bindingdb.org/rwd/bind/info.jsp",
) -> Check:
    name = "bindingdb2026_08_evidence_integrity"
    mirror_path = bindingdb_dir / "bindingdb_source_manifest.json"
    evidence_dir = bindingdb_dir / "evidence_v1"
    evidence_manifest_path = evidence_dir / "manifest.json"
    errors: list[str] = []
    try:
        mirror = json.loads(mirror_path.read_text())
        mirror_source = mirror["source"]
        archive = mirror["archive"]
        extracted = mirror["extracted"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        return Check(name, False, f"invalid BindingDB mirror manifest: {exc}")
    if mirror.get("schema_version") != 1:
        errors.append(f"mirror.schema_version={mirror.get('schema_version')!r}")
    if mirror_source.get("name") != "BindingDB":
        errors.append(f"mirror.source.name={mirror_source.get('name')!r}")
    if mirror.get("release") != expected_release:
        errors.append(f"mirror.release={mirror.get('release')!r}")
    if mirror_source.get("license") != expected_database_license:
        errors.append(f"mirror.source.license={mirror_source.get('license')!r}")
    if mirror_source.get("license_url") != expected_license_url:
        errors.append(f"mirror.source.license_url={mirror_source.get('license_url')!r}")

    validated_mirror_artifacts: dict[str, tuple[Path, str]] = {}
    for label, metadata in (("archive", archive), ("extracted", extracted)):
        if not isinstance(metadata, dict):
            errors.append(f"mirror.{label}=invalid")
            continue
        filename = str(metadata.get("path") or "").strip()
        if not filename or Path(filename).name != filename:
            errors.append(f"mirror.{label}.path={filename!r}")
            continue
        artifact_path = bindingdb_dir / filename
        expected_hash = str(metadata.get("sha256") or "").strip()
        if not artifact_path.exists() or artifact_path.stat().st_size == 0:
            errors.append(f"mirror.{label}=missing")
        elif len(expected_hash) != 64 or _sha256_file(artifact_path) != expected_hash:
            errors.append(f"mirror.{label}=sha256_mismatch")
        else:
            validated_mirror_artifacts[label] = (artifact_path, expected_hash)

    try:
        evidence = json.loads(evidence_manifest_path.read_text())
        evidence_source = evidence["source"]
        inputs = evidence["inputs"]
        artifacts = evidence["artifacts"]
        counts = evidence["counts"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        errors.append(f"invalid evidence manifest:{exc}")
        return Check(name, False, f"errors={errors}")
    if evidence.get("schema_version") != "bindingdb_activity_evidence.v1":
        errors.append(f"evidence.schema_version={evidence.get('schema_version')!r}")
    if evidence_source.get("source_db") != "BindingDB":
        errors.append(f"evidence.source_db={evidence_source.get('source_db')!r}")
    if evidence_source.get("source_release") != expected_release:
        errors.append(f"evidence.source_release={evidence_source.get('source_release')!r}")
    if evidence_source.get("source_license") != expected_evidence_license:
        errors.append(f"evidence.source_license={evidence_source.get('source_license')!r}")
    if evidence_source.get("source_license_url") != expected_license_url:
        errors.append(f"evidence.source_license_url={evidence_source.get('source_license_url')!r}")
    license_policy = evidence_source.get("license_policy")
    if not isinstance(license_policy, dict):
        errors.append("evidence.license_policy=invalid")
    else:
        if license_policy.get("bindingdb_curated") != expected_evidence_license:
            errors.append("evidence.license_policy.bindingdb_curated_mismatch")
        if license_policy.get("chembl_derived") != "CC BY-SA 3.0 (ChEMBL-derived)":
            errors.append("evidence.license_policy.chembl_derived_mismatch")

    extracted_hash = validated_mirror_artifacts.get("extracted", (None, ""))[1]
    if evidence_source.get("sha256") != extracted_hash:
        errors.append("evidence.source.sha256_mismatch")
    mirror_input = inputs.get("mirror_source_manifest") if isinstance(inputs, dict) else None
    if not isinstance(mirror_input, dict):
        errors.append("evidence.inputs.mirror_source_manifest=missing")
    else:
        if mirror_input.get("sha256") != _sha256_file(mirror_path):
            errors.append("evidence.inputs.mirror_source_manifest.sha256_mismatch")
        if mirror_input.get("extracted_sha256") != extracted_hash:
            errors.append("evidence.inputs.mirror_source_manifest.extracted_sha256_mismatch")

    filenames = {
        "activity_evidence": "activity_evidence.parquet",
        "pre_cutoff": "pre_cutoff.parquet",
        "post_cutoff": "post_cutoff.parquet",
    }
    observed_counts: dict[str, int] = {}
    for key, filename in filenames.items():
        metadata = artifacts.get(key) if isinstance(artifacts, dict) else None
        path = evidence_dir / filename
        if not isinstance(metadata, dict):
            errors.append(f"evidence.artifacts.{key}=missing")
            continue
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f"evidence.artifacts.{key}=missing_file")
            continue
        if metadata.get("bytes") != path.stat().st_size:
            errors.append(f"evidence.artifacts.{key}=bytes_mismatch")
        if metadata.get("sha256") != _sha256_file(path):
            errors.append(f"evidence.artifacts.{key}=sha256_mismatch")
        try:
            rows = _parquet_row_count(path)
            expected_rows = int(counts[key])
        except (KeyError, TypeError, ValueError, OSError) as exc:
            errors.append(f"evidence.artifacts.{key}=row_count_error:{exc}")
        else:
            observed_counts[key] = rows
            if rows != expected_rows:
                errors.append(f"evidence.artifacts.{key}=rows:{rows}!={expected_rows}")
    if (
        set(observed_counts) == set(filenames)
        and observed_counts["pre_cutoff"] + observed_counts["post_cutoff"]
        != observed_counts["activity_evidence"]
    ):
        errors.append("evidence.temporal_partition_count_mismatch")
    return Check(name, not errors, f"errors={errors or 'none'}")


def chk_chembl_evidence_integrity(
    chembl_dir: Path,
    *,
    expected_release: str = "37",
    expected_license: str = "CC BY-SA 3.0",
) -> Check:
    name = "chembl37_evidence_integrity"
    manifest_path = chembl_dir / "source_manifest.json"
    try:
        payload = json.loads(manifest_path.read_text())
        source = payload["source"]
        output_hashes = payload["output_sha256"]
        row_counts = payload["row_counts"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        return Check(name, False, f"invalid source manifest: {exc}")
    errors: list[str] = []
    if payload.get("schema_version") != "chembl_activity_evidence.v1":
        errors.append(f"schema_version={payload.get('schema_version')!r}")
    if str(source.get("name", "")).strip() != "ChEMBL":
        errors.append(f"source.name={source.get('name')!r}")
    if str(source.get("release", "")).strip() != expected_release:
        errors.append(f"source.release={source.get('release')!r}")
    if str(source.get("license", "")).strip() != expected_license:
        errors.append(f"source.license={source.get('license')!r}")
    for filename, count_key in (
        ("human_activities.parquet", "human_activities"),
        ("activity_evidence.parquet", "activity_evidence"),
    ):
        path = chembl_dir / filename
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f"{filename}=missing")
            continue
        expected_hash = str(output_hashes.get(filename, "")).strip()
        if not expected_hash or _sha256_file(path) != expected_hash:
            errors.append(f"{filename}=sha256_mismatch")
        try:
            actual_rows = _parquet_row_count(path)
            expected_rows = int(row_counts[count_key])
        except (KeyError, TypeError, ValueError, OSError) as exc:
            errors.append(f"{filename}=row_count_error:{exc}")
        else:
            if actual_rows != expected_rows:
                errors.append(
                    f"{filename}=rows:{actual_rows}!={expected_rows}"
                )
    return Check(name, not errors, f"errors={errors or 'none'}")


def chk_chembl_fingerprint_integrity(
    chembl_dir: Path,
    *,
    expected_release: str = "37",
    expected_license: str = "CC BY-SA 3.0",
) -> Check:
    name = "chembl37_fingerprint_integrity"
    manifest_path = chembl_dir / "fingerprint_manifest.json"
    source_manifest_path = chembl_dir / "source_manifest.json"
    input_path = chembl_dir / "human_activities.parquet"
    artifact_path = chembl_dir / "fp_morgan2_2048.parquet"
    try:
        payload = json.loads(manifest_path.read_text())
        source = payload["source_snapshot"]
        input_meta = payload["input"]
        artifact = payload["artifact"]
        algorithm = payload["algorithm"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        return Check(name, False, f"invalid fingerprint manifest: {exc}")
    errors: list[str] = []
    if payload.get("schema_version") != "chembl_fingerprint_snapshot.v1":
        errors.append(f"schema_version={payload.get('schema_version')!r}")
    if str(source.get("source_release", "")).strip() != expected_release:
        errors.append(f"source_release={source.get('source_release')!r}")
    if str(source.get("source_license", "")).strip() != expected_license:
        errors.append(f"source_license={source.get('source_license')!r}")
    expected_algorithm = {
        "name": "Morgan ECFP4",
        "radius": 2,
        "n_bits": 2048,
        "use_chirality": False,
    }
    for key, value in expected_algorithm.items():
        if algorithm.get(key) != value:
            errors.append(f"algorithm.{key}={algorithm.get(key)!r}")
    for path, expected_hash, label in (
        (source_manifest_path, source.get("manifest_sha256"), "source_manifest"),
        (input_path, input_meta.get("sha256"), "human_activities"),
        (artifact_path, artifact.get("sha256"), "fingerprints"),
    ):
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f"{label}=missing")
            continue
        if not str(expected_hash or "").strip() or _sha256_file(path) != expected_hash:
            errors.append(f"{label}=sha256_mismatch")
    for path, metadata, label in (
        (input_path, input_meta, "human_activities"),
        (artifact_path, artifact, "fingerprints"),
    ):
        if not path.exists():
            continue
        try:
            actual_rows = _parquet_row_count(path)
            expected_rows = int(metadata["rows"])
        except (KeyError, TypeError, ValueError, OSError) as exc:
            errors.append(f"{label}=row_count_error:{exc}")
        else:
            if actual_rows != expected_rows:
                errors.append(f"{label}=rows:{actual_rows}!={expected_rows}")
    return Check(name, not errors, f"errors={errors or 'none'}")


def _validate_manifest_artifact(
    errors: list[str],
    root: Path,
    artifacts: dict,
    artifact_name: str,
    filename: str,
    *,
    rows_key: str | None = None,
) -> None:
    artifact = artifacts.get(artifact_name)
    if not isinstance(artifact, dict):
        errors.append(f"{artifact_name}=missing_manifest_artifact")
        return
    path = root / filename
    if not path.exists() or path.stat().st_size == 0:
        errors.append(f"{artifact_name}=missing")
        return
    expected_hash = str(artifact.get("sha256", "")).strip()
    if not expected_hash or _sha256_file(path) != expected_hash:
        errors.append(f"{artifact_name}=sha256_mismatch")
    expected_bytes = artifact.get("bytes")
    if expected_bytes is not None and expected_bytes != path.stat().st_size:
        errors.append(f"{artifact_name}=bytes:{path.stat().st_size}!={expected_bytes}")
    if rows_key is not None:
        try:
            actual_rows = _parquet_row_count(path)
            expected_rows = int(artifact["rows"])
        except (KeyError, TypeError, ValueError, OSError) as exc:
            errors.append(f"{artifact_name}=row_count_error:{exc}")
        else:
            if actual_rows != expected_rows:
                errors.append(f"{artifact_name}=rows:{actual_rows}!={expected_rows}")


def chk_gtopdb_evidence_integrity(
    gtopdb_dir: Path,
    *,
    expected_release: str = "2026.2",
    expected_license: str = "ODbL 1.0; contents CC BY-SA 4.0",
) -> Check:
    name = "gtopdb2026_2_evidence_integrity"
    source_manifest_path = gtopdb_dir / "source_manifest.json"
    evidence_dir = gtopdb_dir / "evidence_v1"
    evidence_manifest_path = evidence_dir / "manifest.json"
    errors: list[str] = []
    try:
        source_payload = json.loads(source_manifest_path.read_text())
        source = source_payload["source"]
        source_artifacts = source_payload["artifacts"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        return Check(name, False, f"invalid GtoPdb source manifest: {exc}")

    if source_payload.get("schema_version") != "skinscout.gtopdb-source.v1":
        errors.append(f"source.schema_version={source_payload.get('schema_version')!r}")
    if str(source.get("name", "")).strip() != "GtoPdb":
        errors.append(f"source.name={source.get('name')!r}")
    if str(source.get("release", "")).strip() != expected_release:
        errors.append(f"source.release={source.get('release')!r}")
    if str(source.get("license", "")).strip() != expected_license:
        errors.append(f"source.license={source.get('license')!r}")
    if not isinstance(source_artifacts, dict):
        errors.append("source.artifacts=invalid")
        source_artifacts = {}
    for artifact_name in (
        "interactions.csv",
        "ligands.csv",
        "GtP_to_UniProt_mapping.csv",
        "file_descriptions.txt",
        "pubmed_esummary.json",
    ):
        _validate_manifest_artifact(
            errors,
            gtopdb_dir,
            source_artifacts,
            artifact_name,
            artifact_name,
        )

    try:
        evidence_payload = json.loads(evidence_manifest_path.read_text())
        evidence_source = evidence_payload["source"]
        evidence_inputs = evidence_payload["inputs"]
        evidence_artifacts = evidence_payload["artifacts"]
        output_hashes = evidence_payload["output_sha256"]
        row_counts = evidence_payload["row_counts"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        return Check(name, False, f"invalid GtoPdb evidence manifest: {exc}")

    if evidence_payload.get("schema_version") != "gtopdb_activity_evidence.v1":
        errors.append(f"evidence.schema_version={evidence_payload.get('schema_version')!r}")
    if str(evidence_source.get("name", "")).strip() != "GtoPdb":
        errors.append(f"evidence.source.name={evidence_source.get('name')!r}")
    if str(evidence_source.get("release", "")).strip() != expected_release:
        errors.append(f"evidence.source.release={evidence_source.get('release')!r}")
    if str(evidence_source.get("license", "")).strip() != expected_license:
        errors.append(f"evidence.source.license={evidence_source.get('license')!r}")
    if not isinstance(evidence_inputs, dict):
        errors.append("evidence.inputs=invalid")
        evidence_inputs = {}
    source_manifest_input = evidence_inputs.get("source_manifest", {})
    if not isinstance(source_manifest_input, dict):
        errors.append("evidence.inputs.source_manifest=invalid")
    else:
        expected_hash = str(source_manifest_input.get("sha256", "")).strip()
        if not expected_hash or _sha256_file(source_manifest_path) != expected_hash:
            errors.append("source_manifest=sha256_mismatch")
        if source_manifest_input.get("schema_version") != "skinscout.gtopdb-source.v1":
            errors.append(
                f"source_manifest.schema_version={source_manifest_input.get('schema_version')!r}"
            )
    for input_name, filename in (
        ("interactions_csv", "interactions.csv"),
        ("ligands_csv", "ligands.csv"),
        ("target_mapping_csv", "GtP_to_UniProt_mapping.csv"),
        ("pubmed_json", "pubmed_esummary.json"),
    ):
        input_meta = evidence_inputs.get(input_name, {})
        if not isinstance(input_meta, dict):
            errors.append(f"{input_name}=missing_manifest_input")
            continue
        path = gtopdb_dir / filename
        expected_hash = str(input_meta.get("sha256", "")).strip()
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f"{input_name}=missing")
        elif not expected_hash or _sha256_file(path) != expected_hash:
            errors.append(f"{input_name}=sha256_mismatch")
    if not isinstance(evidence_artifacts, dict):
        errors.append("evidence.artifacts=invalid")
        evidence_artifacts = {}
    if not isinstance(output_hashes, dict):
        errors.append("evidence.output_sha256=invalid")
        output_hashes = {}
    if not isinstance(row_counts, dict):
        errors.append("evidence.row_counts=invalid")
        row_counts = {}
    for artifact_name, filename, count_key in (
        ("activity_evidence", "activity_evidence.parquet", "activity_evidence"),
        ("pre_cutoff", "pre_cutoff.parquet", "pre_cutoff"),
        ("post_cutoff", "post_cutoff.parquet", "post_cutoff"),
    ):
        _validate_manifest_artifact(
            errors,
            evidence_dir,
            evidence_artifacts,
            artifact_name,
            filename,
            rows_key=count_key,
        )
        expected_output_hash = str(output_hashes.get(filename, "")).strip()
        artifact = evidence_artifacts.get(artifact_name, {})
        artifact_hash = (
            str(artifact.get("sha256", "")).strip()
            if isinstance(artifact, dict)
            else ""
        )
        if isinstance(artifact, dict) and expected_output_hash != artifact_hash:
            errors.append(f"{artifact_name}=output_sha256_mismatch")
        try:
            artifact_rows = artifact.get("rows") if isinstance(artifact, dict) else None
            if int(row_counts[count_key]) != int(artifact_rows):
                errors.append(f"{artifact_name}=manifest_row_count_mismatch")
        except (KeyError, TypeError, ValueError):
            errors.append(f"{artifact_name}=manifest_row_count_missing")
    return Check(name, not errors, f"errors={errors or 'none'}")


def chk_graphml(path: Path, name: str) -> Check:
    if not path.exists() or path.stat().st_size == 0:
        return Check(name, False, f"{path} missing or empty")
    try:
        import networkx as nx
        graph = nx.read_graphml(path)
    except Exception as exc:  # noqa: BLE001
        return Check(name, False, f"{path} failed to parse: {exc}")
    return Check(name, graph.number_of_nodes() > 0,
                 f"nodes={graph.number_of_nodes()} edges={graph.number_of_edges()}")


def chk_table_claim_quality(
    path: Path,
    name: str,
    required_cols: set[str],
    min_rows: int,
    placeholder_markers: dict[str, set[str]] | None = None,
) -> Check:
    if not path.exists() or path.stat().st_size == 0:
        return Check(name, False, f"{path} missing or empty")
    try:
        import pandas as pd
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
    except Exception as exc:  # noqa: BLE001
        return Check(name, False, f"{path} failed to parse: {exc}")

    missing = sorted(required_cols - set(df.columns))
    markers: list[str] = []
    for col, values in (placeholder_markers or {}).items():
        if col not in df.columns:
            continue
        observed = set(df[col].astype(str))
        hits = sorted(observed & values)
        if hits:
            markers.extend(f"{col}={h}" for h in hits[:5])
    ok = not missing and len(df) >= min_rows and not markers
    detail = (
        f"rows={len(df)} min_rows={min_rows} "
        f"missing_cols={missing or 'none'} placeholders={markers or 'none'}"
    )
    return Check(name, ok, detail)


def chk_kg_claim_quality(path: Path, min_gene_edges: int) -> Check:
    if not path.exists() or path.stat().st_size == 0:
        return Check("claim_skin_efficacy_kg", False, f"{path} missing or empty")
    try:
        import networkx as nx
        graph = nx.read_graphml(path)
    except Exception as exc:  # noqa: BLE001
        return Check("claim_skin_efficacy_kg", False, f"{path} failed to parse: {exc}")
    gene_edges = 0
    pubtator_backed_nodes = 0
    for src, _, attrs in graph.edges(data=True):
        if str(src).startswith("gene:"):
            gene_edges += 1
    for _, attrs in graph.nodes(data=True):
        if attrs.get("pmid_count") or attrs.get("sample_pmids"):
            pubtator_backed_nodes += 1
    ok = gene_edges >= min_gene_edges and pubtator_backed_nodes > 0
    detail = (
        f"nodes={graph.number_of_nodes()} edges={graph.number_of_edges()} "
        f"gene_edges={gene_edges} min_gene_edges={min_gene_edges} "
        f"pubtator_backed_nodes={pubtator_backed_nodes}"
    )
    return Check("claim_skin_efficacy_kg", ok, detail)


def run_all(
    repo: Path,
    strict: bool,
    claim_quality: bool = False,
    check_stage0_flag: bool = True,
    min_cosing_rows: int = 100,
    min_drug_rows: int = 100,
    min_skin_score_rows: int = 1000,
    min_kg_gene_edges: int = 50,
    require_activity_evidence: bool = False,
    require_bindingdb_evidence: bool = True,
    paths: Mapping[str, Path] | None = None,
    config: Mapping[str, Any] | None = None,
    pdbqt_min_success_fraction: float | None = None,
    pdbqt_min_success_count: int | None = None,
) -> int:
    checks = collect_checks(
        repo,
        claim_quality=claim_quality,
        check_stage0_flag=check_stage0_flag,
        min_cosing_rows=min_cosing_rows,
        min_drug_rows=min_drug_rows,
        min_skin_score_rows=min_skin_score_rows,
        min_kg_gene_edges=min_kg_gene_edges,
        require_activity_evidence=require_activity_evidence,
        require_bindingdb_evidence=require_bindingdb_evidence,
        paths=paths,
        config=config,
        pdbqt_min_success_fraction=pdbqt_min_success_fraction,
        pdbqt_min_success_count=pdbqt_min_success_count,
    )
    print(f"{'CHECK':<28}{'OK':<6}DETAIL")
    print("-" * 72)
    n_fail = 0
    for c in checks:
        status = "OK" if c.ok else "FAIL"
        if not c.ok:
            n_fail += 1
        print(f"{c.name:<28}{status:<6}{c.detail}")
    print("-" * 72)
    print(f"Summary: {len(checks) - n_fail}/{len(checks)} passed")
    if n_fail and strict:
        return 1
    return 0


def collect_checks(
    repo: Path,
    *,
    claim_quality: bool = False,
    check_stage0_flag: bool = True,
    min_cosing_rows: int = 100,
    min_drug_rows: int = 100,
    min_skin_score_rows: int = 1000,
    min_kg_gene_edges: int = 50,
    require_activity_evidence: bool = False,
    require_bindingdb_evidence: bool = True,
    paths: Mapping[str, Path] | None = None,
    config: Mapping[str, Any] | None = None,
    pdbqt_min_success_fraction: float | None = None,
    pdbqt_min_success_count: int | None = None,
) -> list[Check]:
    if paths is None and config is not None:
        paths = resolve_stage0_paths(config, root=repo)
    resolved = _collect_paths(repo, paths)
    no_pocket_list = resolved["no_pocket_list"]
    expected_targets = (
        _clean_target_ids(resolved["alphafold_clean"]) if claim_quality else None
    )
    no_pocket_targets = (
        _load_target_list(no_pocket_list)
        if claim_quality
        else None
    )
    expected_pdbqt_targets = (
        _with_pocket_targets(resolved["pockets"]) - (no_pocket_targets or set())
        if claim_quality
        else None
    )
    pdbqt_policy = (
        {
            "min_success_fraction": pdbqt_min_success_fraction,
            "min_success_count": pdbqt_min_success_count,
        }
        if pdbqt_min_success_fraction is not None
        and pdbqt_min_success_count is not None
        else None
    )
    p2rank_check = (
        chk_p2rank(
            resolved["pockets"],
            expected_targets=expected_targets,
            no_pocket_targets=no_pocket_targets,
        )
        if claim_quality
        else chk_p2rank(resolved["pockets"])
    )
    pdbqt_check = (
        chk_pdbqt_roundtrip(
            resolved["pdbqt"],
            expected_targets=expected_pdbqt_targets,
            manifest_path=_pdbqt_manifest_path(resolved["pdbqt"]),
            expected_policy=pdbqt_policy,
        )
        if claim_quality
        else chk_pdbqt_roundtrip(resolved["pdbqt"])
    )
    skin_score = resolved["skin_expression"] / "skin_score.tsv"
    canonical_fasta = resolved["uniprot_human_fasta"]
    checks = [
        chk_alphafold_count(resolved["alphafold_raw"]),
        chk_cleaned_count(resolved["alphafold_clean"]),
        p2rank_check,
        pdbqt_check,
        chk_mmseqs(resolved["mmseqs"]),
        chk_table(skin_score, "v3_skin_score", {"uniprot", "skin_score"}),
        chk_skin_score_axes(skin_score, "v3_skin_score_axes"),
        chk_table(resolved["cosing"] / "cosing.parquet",
                  "v3_cosing", {"inci_name"}),
        chk_table(resolved["drug_avoidance"] / "drugs.parquet",
                  "v3_drug_avoidance", {"smiles"}),
        chk_graphml(resolved["skin_kg"] / "skin_efficacy.graphml",
                    "v3_skin_efficacy_kg"),
        chk_nonempty_file(resolved["skin_proteome"] / "skin_proteome.tsv",
                          "v3_skin_proteome"),
    ]
    if require_activity_evidence:
        checks.extend([
            chk_parquet_schema(
                resolved["chembl"] / "activity_evidence.parquet",
                "chembl37_activity_evidence",
                {
                    "source_db",
                    "source_release",
                    "source_license",
                    "activity_id",
                    "assay_id",
                    "assay_confidence_score",
                    "document_year",
                    "standard_relation",
                    "pchembl",
                    "uniprot",
                    "smiles",
                },
            ),
            chk_parquet_schema(
                resolved["chembl"] / "fp_morgan2_2048.parquet",
                "chembl37_fingerprints",
                {"molecule_chembl_id", "bitvec"},
            ),
            chk_evidence_manifest(
                resolved["chembl"] / "source_manifest.json",
                "chembl37_source_manifest",
                {"schema_version", "source", "input_sha256", "output_sha256"},
            ),
            chk_chembl_evidence_integrity(resolved["chembl"]),
            chk_chembl_fingerprint_integrity(resolved["chembl"]),
            chk_discovery_alias_integrity(repo),
            chk_parquet_schema(
                resolved["gtopdb"] / "evidence_v1" / "activity_evidence.parquet",
                "gtopdb_activity_evidence",
                {
                    "evidence_id",
                    "source_db",
                    "source_origin",
                    "source_release",
                    "source_license",
                    "chembl_derived_license_flag",
                    "ligand_smiles",
                    "ligand_inchikey",
                    "ligand_id",
                    "uniprot",
                    "single_chain_target",
                    "organism",
                    "affinity_type",
                    "relation",
                    "censor",
                    "affinity_value",
                    "affinity_unit",
                    "source_pmid",
                    "publication_date",
                    "evidence_date",
                    "evidence_date_source",
                    "temporal_split",
                    "gtopdb_target_id",
                    "gtopdb_ligand_id",
                    "supporting_pmids",
                    "supporting_pmids_sha256",
                },
            ),
            chk_parquet_schema(
                resolved["gtopdb"] / "evidence_v1" / "pre_cutoff.parquet",
                "gtopdb_pre_cutoff_evidence",
                {"evidence_id", "source_db", "temporal_split"},
                min_rows=0,
            ),
            chk_parquet_schema(
                resolved["gtopdb"] / "evidence_v1" / "post_cutoff.parquet",
                "gtopdb_post_cutoff_evidence",
                {"evidence_id", "source_db", "temporal_split"},
                min_rows=0,
            ),
            chk_evidence_manifest(
                resolved["gtopdb"] / "source_manifest.json",
                "gtopdb_source_manifest",
                {"schema_version", "source", "artifacts"},
            ),
            chk_evidence_manifest(
                resolved["gtopdb"] / "evidence_v1" / "manifest.json",
                "gtopdb_evidence_manifest",
                {
                    "schema_version",
                    "source",
                    "inputs",
                    "policy",
                    "row_counts",
                    "artifacts",
                    "output_sha256",
                },
            ),
            chk_gtopdb_evidence_integrity(resolved["gtopdb"]),
        ])
    if require_activity_evidence and require_bindingdb_evidence:
        checks.extend([
            chk_parquet_schema(
                resolved["bindingdb"] / "evidence_v1" / "activity_evidence.parquet",
                "bindingdb_activity_evidence",
                {
                    "source_db",
                    "source_release",
                    "source_license",
                    "evidence_id",
                    "evidence_date",
                    "affinity_type",
                    "relation",
                    "uniprot",
                },
            ),
            chk_evidence_manifest(
                resolved["bindingdb"] / "evidence_v1" / "manifest.json",
                "bindingdb_evidence_manifest",
                {"schema_version", "source", "inputs", "policy", "artifacts"},
            ),
            chk_bindingdb_evidence_integrity(resolved["bindingdb"]),
            chk_parquet_schema(
                resolved["evidence_splits"] / "post_cutoff_novel_pairs.parquet",
                "temporal_novel_pair_evidence",
                {
                    "evidence_id",
                    "evidence_date",
                    "pair_seen_pre_cutoff",
                    "molecule_seen_pre_cutoff",
                    "target_seen_pre_cutoff",
                },
            ),
            chk_evidence_manifest(
                resolved["evidence_splits"] / "split_manifest.json",
                "temporal_split_manifest",
                {"schema_version", "cutoff_date", "inputs", "policy", "outputs"},
            ),
        ])
    if check_stage0_flag:
        checks.insert(5, chk_flag(resolved["manifests"]))
    if claim_quality:
        checks.extend([
            chk_target_list_file(no_pocket_list),
            chk_canonical_sequences(
                canonical_fasta,
                canonical_fasta.with_name(f"{canonical_fasta.name}.manifest.json"),
                resolved["alphafold_clean"],
            ),
            chk_mmseqs_training_cutoff(resolved["mmseqs"]),
            chk_table_claim_quality(
                skin_score,
                "claim_skin_score",
                {"uniprot", "skin_score"},
                min_skin_score_rows,
            ),
            chk_skin_score_axes(skin_score, "claim_skin_score_axes"),
            chk_table_claim_quality(
                resolved["cosing"] / "cosing.parquet",
                "claim_cosing_reference",
                {"inci_name", "smiles", "inchikey", "ecfp4", "scaffold_smiles"},
                min_cosing_rows,
                placeholder_markers={
                    "inci_name": {"Niacinamide", "Retinol", "Glycerin"},
                },
            ),
            chk_table_claim_quality(
                resolved["drug_avoidance"] / "drugs.parquet",
                "claim_drug_reference",
                {"drug_id", "name", "smiles", "inchikey", "ecfp4"},
                min_drug_rows,
                placeholder_markers={
                    "drug_id": {"PLACEHOLDER", "PLACEHOLDER1", "PLACEHOLDER2"},
                    "name": {"no_drug_db_yet", "Aspirin", "Ibuprofen"},
                },
            ),
            chk_table_claim_quality(
                resolved["drug_avoidance"] / "scaffolds.parquet",
                "claim_drug_scaffolds",
                {"scaffold_smiles", "n_drugs"},
                min(20, min_drug_rows),
            ),
            chk_kg_claim_quality(
                resolved["skin_kg"] / "skin_efficacy.graphml",
                min_kg_gene_edges,
            ),
        ])
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", type=Path)
    parser.add_argument("--strict", action="store_true",
                        help="compatibility alias; strict is the default")
    parser.add_argument(
        "--allow-soft-failures",
        action="store_true",
        help="exit zero despite failed checks only for explicit diagnostics",
    )
    parser.add_argument(
        "--claim-quality",
        action="store_true",
        help="add stricter non-placeholder reference checks for SOTA claims",
    )
    parser.add_argument(
        "--skip-stage0-flag",
        action="store_true",
        help="skip the final stage0_complete.flag existence check while creating it",
    )
    parser.add_argument("--min-cosing-rows", type=int, default=100)
    parser.add_argument("--min-drug-rows", type=int, default=100)
    parser.add_argument("--min-skin-score-rows", type=int, default=1000)
    parser.add_argument("--min-kg-gene-edges", type=int, default=50)
    parser.add_argument(
        "--require-activity-evidence",
        action="store_true",
        help=(
            "require ChEMBL, BindingDB, and GtoPdb evidence snapshots "
            "and temporal split artifacts"
        ),
    )
    parser.add_argument(
        "--allow-bindingdb-placeholder",
        action="store_true",
        help="skip BindingDB and combined evidence checks for explicit diagnostics",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Workflow YAML path; defaults to REPO/workflow/config.yaml",
    )
    parser.add_argument(
        "--extra-config",
        action="append",
        default=[],
        help="Dotted config override, key=value; repeatable",
    )
    parser.add_argument(
        "--pdbqt-min-success-fraction",
        type=float,
        default=None,
        help="Producer policy the receptor manifest must meet",
    )
    parser.add_argument(
        "--pdbqt-min-success-count",
        type=int,
        default=None,
        help="Producer minimum success count the manifest must meet",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    repo = args.repo.resolve()
    try:
        config = apply_config_overrides(
            load_workflow_config(args.config, repo=repo),
            args.extra_config,
        )
    except (OSError, ValueError) as exc:
        print(f"Stage 0 verifier config override failed: {exc}", file=sys.stderr)
        sys.exit(2)
    paths = resolve_stage0_paths(config, root=repo)
    sys.exit(run_all(
        repo,
        strict=not args.allow_soft_failures,
        claim_quality=args.claim_quality,
        check_stage0_flag=not args.skip_stage0_flag,
        min_cosing_rows=args.min_cosing_rows,
        min_drug_rows=args.min_drug_rows,
        min_skin_score_rows=args.min_skin_score_rows,
        min_kg_gene_edges=args.min_kg_gene_edges,
        require_activity_evidence=args.require_activity_evidence,
        require_bindingdb_evidence=not args.allow_bindingdb_placeholder,
        paths=paths,
        pdbqt_min_success_fraction=args.pdbqt_min_success_fraction,
        pdbqt_min_success_count=args.pdbqt_min_success_count,
    ))


if __name__ == "__main__":
    main()
