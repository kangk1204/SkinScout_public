#!/usr/bin/env python3
"""Build an evaluation-panel-independent human target candidate universe."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from build_target_cluster_map import (
    ALLOWED_EXCLUSION_REASONS,
    _cluster_assignments,
    _fasta_records,
)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from activity_recovery_contracts import (  # noqa: E402
    ALPHAFOLD_HUMAN_V4_SOURCE,
    EXPECTED_ALPHAFOLD_BASE_TARGET_COUNT,
    EXPECTED_SCREENABLE_TARGET_COUNT,
)


SCHEMA_VERSION = "skinscout.screenable-target-cluster-map.v2"
BASE_MAP_SCHEMA = "skinscout.target-cluster-map.v2"
EVIDENCE_MAP_SCHEMA = "skinscout.target-cluster-map.v2"
EVIDENCE_SEQUENCE_SCHEMA = "skinscout.evidence-target-sequence-universe.v2"
SOURCE_MANIFEST_SCHEMA = "skinscout.protein-sequence-source.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_file(path: Path, label: str) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"{label} is required and must be non-empty: {path}")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _require_file(path, label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Unable to parse {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} root must be an object: {path}")
    return payload


def _load_registered_map(
    csv_path: Path,
    manifest_path: Path,
    *,
    label: str,
    schema: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = _read_json(manifest_path, f"{label} manifest")
    if manifest.get("schema_version") != schema:
        raise SystemExit(f"{label} manifest schema must be {schema}: {manifest_path}")
    _require_file(csv_path, f"{label} CSV")
    frame = pd.read_csv(csv_path)
    required = {"uniprot", "target_cluster_30", "target_cluster_50"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"{label} CSV missing columns {missing}: {csv_path}")
    for column in required:
        values = frame[column].fillna("").astype(str).str.strip()
        if values.eq("").any():
            raise SystemExit(f"{label} CSV contains blank {column}: {csv_path}")
        frame[column] = values
    if frame["uniprot"].duplicated().any():
        raise SystemExit(f"{label} CSV contains duplicate UniProt accessions")
    artifact = manifest.get("artifact")
    if not isinstance(artifact, dict):
        raise SystemExit(f"{label} manifest lacks artifact registration")
    if artifact.get("sha256") != _sha256(csv_path) or int(
        artifact.get("rows", -1)
    ) != len(frame):
        raise SystemExit(f"{label} CSV does not match its manifest")
    return frame, manifest


def _fasta_map(path: Path) -> dict[str, str]:
    records = _fasta_records(path)
    output: dict[str, str] = {}
    for record in records:
        if record.accession in output:
            raise SystemExit(f"Duplicate FASTA accession {record.accession!r}: {path}")
        output[record.accession] = record.sequence
    return output


def _validate_base_fasta(
    base_ids: set[str],
    base_manifest: dict[str, Any],
    base_manifest_path: Path,
) -> tuple[Path, dict[str, str], dict[str, str], dict[str, Any]]:
    inputs = base_manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise SystemExit("base target manifest lacks inputs")
    path_text = str(inputs.get("fasta") or "").strip()
    expected_sha = str(inputs.get("fasta_sha256") or "").strip()
    if not path_text or not expected_sha:
        raise SystemExit("base target manifest lacks FASTA path/hash provenance")
    fasta_path = Path(path_text).resolve()
    _require_file(fasta_path, "base target FASTA")
    if _sha256(fasta_path) != expected_sha:
        raise SystemExit("base target FASTA sha256 mismatch")
    records = _fasta_map(fasta_path)
    if set(records) != base_ids:
        raise SystemExit("base target CSV accessions do not exactly match base FASTA")
    source_record = inputs.get("source_manifest")
    if not isinstance(source_record, dict):
        raise SystemExit("base target manifest lacks independent source provenance")
    raw_source_path = str(source_record.get("path") or "").strip()
    if not raw_source_path:
        raise SystemExit("base target manifest source provenance lacks path")
    source_path = Path(raw_source_path)
    if not source_path.is_absolute():
        source_path = base_manifest_path.parent / source_path
    source_path = source_path.resolve()
    _require_file(source_path, "base target source manifest")
    if source_record.get("sha256") != _sha256(source_path):
        raise SystemExit("base target source manifest sha256 mismatch")
    source_manifest = _read_json(source_path, "base target source manifest")
    if source_manifest.get("schema_version") != SOURCE_MANIFEST_SCHEMA:
        raise SystemExit("base target source manifest schema mismatch")
    source = source_manifest.get("source")
    if source != ALPHAFOLD_HUMAN_V4_SOURCE or source_record.get("source") != source:
        raise SystemExit(
            "base target source must be the pinned AlphaFold human proteome v4 snapshot"
        )
    source_artifact = source_manifest.get("sequence_artifact")
    recorded_artifact = source_record.get("sequence_artifact")
    if not isinstance(source_artifact, dict) or not isinstance(recorded_artifact, dict):
        raise SystemExit("base target source manifest lacks sequence artifact provenance")
    if source_artifact.get("sequence_role") != "canonical_uniprot_reference_sequence":
        raise SystemExit(
            "base target FASTA must contain canonical UniProt reference sequences; "
            "structure-observed or pLDDT-filtered sequences are invalid"
        )
    source_fasta = Path(str(source_artifact.get("path") or ""))
    if not source_fasta.is_absolute():
        source_fasta = source_path.parent / source_fasta
    try:
        source_count = int(source_artifact.get("accession_count", -1))
    except (TypeError, ValueError) as exc:
        raise SystemExit("base target source accession count is invalid") from exc
    if (
        source_fasta.resolve() != fasta_path
        or source_artifact.get("sha256") != expected_sha
        or source_count != len(base_ids)
        or recorded_artifact.get("sha256") != expected_sha
        or int(recorded_artifact.get("accession_count", -1)) != len(base_ids)
    ):
        raise SystemExit("base target source artifact does not match the base FASTA")
    return fasta_path, records, source, {
        "path": str(source_path),
        "sha256": _sha256(source_path),
        "schema_version": SOURCE_MANIFEST_SCHEMA,
        "sequence_role": source_artifact["sequence_role"],
        "derivation": source_artifact.get("derivation"),
    }


def _validate_evidence_provenance(
    evidence_ids: set[str], evidence_manifest: dict[str, Any]
) -> tuple[Path, dict[str, str], dict[str, Any]]:
    inputs = evidence_manifest.get("inputs")
    sequence_record = (
        inputs.get("sequence_universe_manifest") if isinstance(inputs, dict) else None
    )
    if not isinstance(sequence_record, dict):
        raise SystemExit("evidence target manifest lacks sequence-universe provenance")
    if sequence_record.get("schema_version") != EVIDENCE_SEQUENCE_SCHEMA:
        raise SystemExit("evidence target sequence-universe schema is not supported")
    if int(sequence_record.get("accepted_count", -1)) != len(evidence_ids):
        raise SystemExit("evidence target count does not match sequence-universe provenance")
    source = sequence_record.get("uniprot_source")
    if not isinstance(source, dict):
        raise SystemExit("evidence target manifest lacks UniProt source provenance")
    if (
        source.get("name") != "UniProtKB"
        or not str(source.get("release") or "").strip()
        or source.get("license") != "CC BY 4.0"
    ):
        raise SystemExit("evidence target UniProt release/license provenance is invalid")
    path_text = str(sequence_record.get("target_fasta_path") or "").strip()
    expected_sha = str(sequence_record.get("target_fasta_sha256") or "").strip()
    if not path_text or not expected_sha:
        raise SystemExit("evidence target manifest lacks target FASTA path/hash")
    fasta_path = Path(path_text).resolve()
    _require_file(fasta_path, "evidence target FASTA")
    if _sha256(fasta_path) != expected_sha:
        raise SystemExit("evidence target FASTA sha256 mismatch")
    records = _fasta_map(fasta_path)
    if set(records) != evidence_ids:
        raise SystemExit("evidence target CSV accessions do not exactly match target FASTA")
    return fasta_path, records, source


def _validated_evidence_exclusions(
    evidence_manifest: dict[str, Any],
    evidence_ids: set[str],
) -> dict[str, Any]:
    payload = evidence_manifest.get("excluded_targets")
    if not isinstance(payload, dict):
        raise SystemExit("evidence target manifest lacks audited excluded_targets")
    records = payload.get("records")
    if not isinstance(records, list) or payload.get("count") != len(records):
        raise SystemExit("evidence target excluded target count is invalid")
    policy = str(payload.get("policy") or "").strip()
    if not policy:
        raise SystemExit("evidence target exclusion policy must be nonblank")
    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise SystemExit(f"evidence target exclusion {index} must be an object")
        accession = str(record.get("uniprot") or "").strip()
        reason = str(record.get("reason") or "").strip()
        if not accession or reason not in ALLOWED_EXCLUSION_REASONS:
            raise SystemExit(f"evidence target exclusion {index} is invalid")
        if accession in seen:
            raise SystemExit(f"duplicate evidence target exclusion: {accession}")
        if accession in evidence_ids:
            raise SystemExit(
                f"excluded evidence target unexpectedly has a cluster assignment: {accession}"
            )
        seen.add(accession)
        validated.append(dict(record))
    return {
        "count": len(validated),
        "policy": (
            "Audited exclusions inherited from the provenance-bound evidence target "
            "sequence universe override general screenable-map assignments for benchmark "
            "evidence."
        ),
        "source_policy": policy,
        "records": validated,
    }


def _write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["uniprot", "target_cluster_30", "target_cluster_50"],
        )
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def build(args: argparse.Namespace) -> dict[str, Any]:
    for output in (args.out_csv, args.out_manifest):
        output.unlink(missing_ok=True)
        output.with_suffix(output.suffix + ".tmp").unlink(missing_ok=True)
    try:
        base_frame, base_manifest = _load_registered_map(
            args.base_target_csv,
            args.base_target_manifest,
            label="base target map",
            schema=BASE_MAP_SCHEMA,
        )
        evidence_frame, evidence_manifest = _load_registered_map(
            args.evidence_target_csv,
            args.evidence_target_manifest,
            label="evidence target map",
            schema=EVIDENCE_MAP_SCHEMA,
        )
        base_ids = set(base_frame["uniprot"])
        evidence_ids = set(evidence_frame["uniprot"])
        evidence_exclusions = _validated_evidence_exclusions(
            evidence_manifest,
            evidence_ids,
        )
        base_fasta, _, base_source, base_source_record = _validate_base_fasta(
            base_ids,
            base_manifest,
            args.base_target_manifest,
        )
        evidence_fasta, evidence_records, uniprot_source = (
            _validate_evidence_provenance(evidence_ids, evidence_manifest)
        )
        supplemental_records = _fasta_map(args.supplemental_fasta)
        expected_supplemental = evidence_ids - base_ids
        if set(supplemental_records) != expected_supplemental:
            missing = sorted(expected_supplemental - set(supplemental_records))
            extra = sorted(set(supplemental_records) - expected_supplemental)
            raise SystemExit(
                "supplemental FASTA must exactly equal evidence targets absent from the "
                f"independent base universe; missing={missing[:10]} extra={extra[:10]}"
            )
        for accession, sequence in supplemental_records.items():
            if evidence_records[accession] != sequence:
                raise SystemExit(
                    f"supplemental FASTA sequence differs from UniProt evidence sequence: {accession}"
                )

        target_ids = sorted(base_ids | evidence_ids)
        production_counts_pass = (
            len(base_ids) == EXPECTED_ALPHAFOLD_BASE_TARGET_COUNT
            and len(target_ids) == EXPECTED_SCREENABLE_TARGET_COUNT
        )
        if not args.fixture_mode and not production_counts_pass:
            raise SystemExit(
                "production screenable target universe requires exactly "
                f"{EXPECTED_ALPHAFOLD_BASE_TARGET_COUNT} AlphaFold base targets and "
                f"{EXPECTED_SCREENABLE_TARGET_COUNT} union targets; observed "
                f"base={len(base_ids)} union={len(target_ids)}"
            )
        target_set = set(target_ids)
        cluster30 = _cluster_assignments(
            args.cluster30_tsv,
            label="screenable MMseqs2 30% cluster TSV",
            fasta_ids=target_set,
        )
        cluster50 = _cluster_assignments(
            args.cluster50_tsv,
            label="screenable MMseqs2 50% cluster TSV",
            fasta_ids=target_set,
        )
        rows = [
            {
                "uniprot": accession,
                "target_cluster_30": cluster30[accession],
                "target_cluster_50": cluster50[accession],
            }
            for accession in target_ids
        ]
        _write_csv(rows, args.out_csv)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "universe_policy": {
                "candidate_definition": (
                    "union of the independent AlphaFold human-proteome receptor snapshot "
                    "and claim-grade activity-evidence targets absent from that snapshot"
                ),
                "evaluation_panel_used": False,
                "known_target_assistance": False,
                "base_target_count": len(base_ids),
                "evidence_target_count": len(evidence_ids),
                "supplemental_target_count": len(expected_supplemental),
                "union_target_count": len(target_ids),
            },
            "production_contract": {
                "profile": "alphafold-human-v4-plus-claim-grade-evidence",
                "fixture_mode": bool(args.fixture_mode),
                "expected_base_target_count": EXPECTED_ALPHAFOLD_BASE_TARGET_COUNT,
                "expected_union_target_count": EXPECTED_SCREENABLE_TARGET_COUNT,
                "passes": bool(production_counts_pass and not args.fixture_mode),
            },
            "excluded_targets": evidence_exclusions,
            "sources": {
                "independent_base": base_source,
                "evidence_sequence_source": uniprot_source,
            },
            "parameters": {
                "engine": "MMseqs2 easy-cluster",
                "mmseqs_version": args.mmseqs_version,
                "cluster_mode": "connected-component",
                "single_step_clustering": True,
                "coverage": 0.8,
                "coverage_mode": 0,
                "sensitivity": 7.5,
                "max_seqs": 1000,
                "sequence_identity_thresholds": {"main": 0.3, "sensitivity": 0.5},
            },
            "inputs": {
                "base_target_csv": {
                    "path": str(args.base_target_csv.resolve()),
                    "sha256": _sha256(args.base_target_csv),
                    "manifest_path": str(args.base_target_manifest.resolve()),
                    "manifest_sha256": _sha256(args.base_target_manifest),
                    "fasta_path": str(base_fasta),
                    "fasta_sha256": _sha256(base_fasta),
                    "source_manifest": base_source_record,
                },
                "evidence_target_csv": {
                    "path": str(args.evidence_target_csv.resolve()),
                    "sha256": _sha256(args.evidence_target_csv),
                    "manifest_path": str(args.evidence_target_manifest.resolve()),
                    "manifest_sha256": _sha256(args.evidence_target_manifest),
                    "fasta_path": str(evidence_fasta),
                    "fasta_sha256": _sha256(evidence_fasta),
                },
                "supplemental_fasta": {
                    "path": str(args.supplemental_fasta.resolve()),
                    "sha256": _sha256(args.supplemental_fasta),
                    "accession_count": len(supplemental_records),
                },
                "cluster30_tsv": {
                    "path": str(args.cluster30_tsv.resolve()),
                    "sha256": _sha256(args.cluster30_tsv),
                },
                "cluster50_tsv": {
                    "path": str(args.cluster50_tsv.resolve()),
                    "sha256": _sha256(args.cluster50_tsv),
                },
            },
            "artifact": {
                "path": str(args.out_csv.resolve()),
                "sha256": _sha256(args.out_csv),
                "rows": len(rows),
                "cluster30_count": len(set(cluster30.values())),
                "cluster50_count": len(set(cluster50.values())),
            },
        }
        _write_json(manifest, args.out_manifest)
        return manifest
    except BaseException:
        for output in (args.out_csv, args.out_manifest):
            output.unlink(missing_ok=True)
            output.with_suffix(output.suffix + ".tmp").unlink(missing_ok=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-target-csv", required=True, type=Path)
    parser.add_argument("--base-target-manifest", required=True, type=Path)
    parser.add_argument("--evidence-target-csv", required=True, type=Path)
    parser.add_argument("--evidence-target-manifest", required=True, type=Path)
    parser.add_argument("--supplemental-fasta", required=True, type=Path)
    parser.add_argument("--cluster30-tsv", required=True, type=Path)
    parser.add_argument("--cluster50-tsv", required=True, type=Path)
    parser.add_argument("--mmseqs-version", required=True)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-manifest", required=True, type=Path)
    parser.add_argument(
        "--fixture-mode",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    if not args.mmseqs_version.strip():
        raise SystemExit("--mmseqs-version must be nonblank")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build(args)
    print(
        "[screenable-targets] wrote "
        f"targets={manifest['artifact']['rows']} "
        f"supplemental={manifest['universe_policy']['supplemental_target_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
