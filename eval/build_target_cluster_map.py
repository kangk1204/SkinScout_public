#!/usr/bin/env python3
"""Bind MMseqs2 cluster outputs to a complete, auditable UniProt map."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = "skinscout.target-cluster-map.v2"
SEQUENCE_UNIVERSE_SCHEMA_VERSION = "skinscout.evidence-target-sequence-universe.v2"
SOURCE_MANIFEST_SCHEMA_VERSION = "skinscout.protein-sequence-source.v1"
ALLOWED_EXCLUSION_REASONS = {
    "not_found_in_uniprot_release",
    "unresolved_or_obsolete_uniprot_record",
    "taxonomy_mismatch",
    "sequence_below_minimum_length",
}


@dataclass(frozen=True)
class FastaRecord:
    accession: str
    sequence: str


@dataclass
class FastaProvenance:
    role: str
    path: Path
    sha256: str
    accepted_count: int = 0
    rejected_count: int = 0

    def to_manifest(self) -> dict[str, object]:
        return {
            "role": self.role,
            "path": str(self.path.resolve()),
            "sha256": self.sha256,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_file(path: Path, label: str) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"{label} is required and must be non-empty: {path}")


def _fasta_records(path: Path) -> list[FastaRecord]:
    _require_file(path, "Protein FASTA")
    records: list[FastaRecord] = []
    current_accession: str | None = None
    current_sequence: list[str] = []

    def flush_record() -> None:
        nonlocal current_accession, current_sequence
        if current_accession is None:
            return
        records.append(
            FastaRecord(
                accession=current_accession,
                sequence="".join(current_sequence).upper(),
            )
        )
        current_accession = None
        current_sequence = []

    with path.open() as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if raw.startswith(">"):
                flush_record()
                accession = raw[1:].strip().split(maxsplit=1)[0]
                if not accession:
                    raise SystemExit(
                        f"Blank FASTA identifier at line {line_number}: {path}"
                    )
                current_accession = accession
                continue
            if current_accession is None:
                if line:
                    raise SystemExit(
                        f"Protein FASTA sequence appears before an identifier at line "
                        f"{line_number}: {path}"
                    )
                continue
            current_sequence.append(line)
    flush_record()
    if not records:
        raise SystemExit(f"Protein FASTA contains no identifiers: {path}")
    return records


def _merged_fasta_accessions(
    fasta_paths: list[tuple[str, Path]],
    *,
    min_sequence_length: int,
) -> tuple[list[str], list[dict[str, object]]]:
    seen_sequences: dict[str, str] = {}
    accepted: set[str] = set()
    provenance: list[FastaProvenance] = []

    for role, path in fasta_paths:
        records = _fasta_records(path)
        file_provenance = FastaProvenance(role=role, path=path, sha256=_sha256(path))
        for record in records:
            existing = seen_sequences.get(record.accession)
            if existing is not None:
                if existing != record.sequence:
                    raise SystemExit(
                        f"Conflicting FASTA sequence for accession "
                        f"{record.accession!r}: {path}"
                    )
                if len(record.sequence) < min_sequence_length:
                    file_provenance.rejected_count += 1
                else:
                    file_provenance.accepted_count += 1
                continue

            seen_sequences[record.accession] = record.sequence
            if len(record.sequence) < min_sequence_length:
                file_provenance.rejected_count += 1
                continue

            accepted.add(record.accession)
            file_provenance.accepted_count += 1
        provenance.append(file_provenance)

    if not accepted:
        raise SystemExit(
            f"Protein FASTA contains no sequences meeting --min-sequence-length "
            f"{min_sequence_length}"
        )
    return sorted(accepted), [item.to_manifest() for item in provenance]


def _cluster_assignments(
    path: Path,
    *,
    label: str,
    fasta_ids: set[str],
) -> dict[str, str]:
    _require_file(path, label)
    assignments: dict[str, str] = {}
    with path.open() as handle:
        for line_number, raw in enumerate(handle, start=1):
            fields = raw.rstrip("\n").split("\t")
            if len(fields) != 2 or not fields[0].strip() or not fields[1].strip():
                raise SystemExit(
                    f"{label} requires exactly two nonblank TSV columns at line "
                    f"{line_number}: {path}"
                )
            representative, member = (field.strip() for field in fields)
            unknown = sorted({representative, member} - fasta_ids)
            if unknown:
                raise SystemExit(
                    f"{label} contains identifier(s) absent from FASTA at line "
                    f"{line_number}: {', '.join(unknown)}"
                )
            if member in assignments:
                raise SystemExit(
                    f"{label} assigns member {member!r} more than once: {path}"
                )
            assignments[member] = representative
    missing = sorted(fasta_ids - set(assignments))
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        raise SystemExit(
            f"{label} is incomplete; {len(missing)} FASTA identifiers lack a cluster "
            f"assignment: {preview}{suffix}"
        )
    return assignments


def _write_csv_atomic(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["uniprot", "target_cluster_30", "target_cluster_50"],
        )
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _write_json_atomic(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _sequence_universe_provenance(
    path: Path | None,
    *,
    base_fasta: Path,
    additional_fastas: list[Path],
    fasta_ids: set[str],
    min_sequence_length: int,
) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    if path is None:
        return None, []
    _require_file(path, "Target sequence universe manifest")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid target sequence universe manifest JSON: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != SEQUENCE_UNIVERSE_SCHEMA_VERSION:
        raise SystemExit(f"Unsupported target sequence universe manifest schema: {path}")

    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    target_fasta = (
        artifacts.get("target_fasta")
        if isinstance(artifacts.get("target_fasta"), dict)
        else {}
    )
    target_path_text = str(target_fasta.get("path") or "").strip()
    target_sha256 = str(target_fasta.get("sha256") or "").strip()
    if not target_path_text or not target_sha256:
        raise SystemExit("Target sequence universe manifest lacks target FASTA provenance")
    target_path = Path(target_path_text).resolve()
    supplied_fastas = {base_fasta.resolve(), *(item.resolve() for item in additional_fastas)}
    if target_path not in supplied_fastas:
        raise SystemExit(
            "Target sequence universe FASTA must be supplied with --fasta or --additional-fasta"
        )
    _require_file(target_path, "Target sequence universe FASTA")
    if _sha256(target_path) != target_sha256:
        raise SystemExit("Target sequence universe FASTA sha256 mismatch")

    observed_targets = {
        record.accession
        for record in _fasta_records(target_path)
        if len(record.sequence) >= min_sequence_length
    }
    accepted_raw = target_fasta.get("accepted_accessions")
    if not isinstance(accepted_raw, list) or any(
        not isinstance(accession, str) or not accession.strip() for accession in accepted_raw
    ):
        raise SystemExit("Target sequence universe accepted_accessions must be a nonblank list")
    accepted = {accession.strip() for accession in accepted_raw}
    if accepted != observed_targets:
        raise SystemExit(
            "Target sequence universe accepted_accessions do not match target FASTA"
        )
    if accepted != fasta_ids:
        raise SystemExit(
            "Merged FASTA accessions must exactly match the provenance-bound evidence target universe"
        )

    exclusions_payload = (
        payload.get("exclusions") if isinstance(payload.get("exclusions"), dict) else {}
    )
    records_raw = exclusions_payload.get("records")
    if not isinstance(records_raw, list):
        raise SystemExit("Target sequence universe exclusions.records must be a list")
    exclusions: list[dict[str, object]] = []
    seen_exclusions: set[str] = set()
    for index, record in enumerate(records_raw):
        if not isinstance(record, dict):
            raise SystemExit(f"Target sequence exclusion at index {index} must be an object")
        accession = str(record.get("uniprot") or "").strip()
        reason = str(record.get("reason") or "").strip()
        if not accession or reason not in ALLOWED_EXCLUSION_REASONS:
            raise SystemExit(f"Invalid target sequence exclusion at index {index}")
        if accession in seen_exclusions:
            raise SystemExit(f"Duplicate target sequence exclusion: {accession}")
        if accession in fasta_ids:
            raise SystemExit(
                f"Excluded target sequence accession is present in merged FASTA: {accession}"
            )
        seen_exclusions.add(accession)
        exclusions.append(dict(record))

    evidence_targets = (
        payload.get("evidence_targets")
        if isinstance(payload.get("evidence_targets"), dict)
        else {}
    )
    evidence_target_count = evidence_targets.get("total_count")
    if evidence_target_count != len(accepted) + len(exclusions):
        raise SystemExit(
            "Target sequence universe target count does not equal accepted plus excluded targets"
        )
    if exclusions_payload.get("count") != len(exclusions):
        raise SystemExit("Target sequence universe exclusion count mismatch")

    provenance = {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "schema_version": payload["schema_version"],
        "target_fasta_path": str(target_path),
        "target_fasta_sha256": target_sha256,
        "accepted_count": len(accepted),
        "excluded_count": len(exclusions),
        "evidence_target_count": evidence_targets.get("total_count"),
        "uniprot_source": (
            payload.get("inputs", {}).get("uniprot_metadata", {}).get("source", {})
            if isinstance(payload.get("inputs"), dict)
            else {}
        ),
    }
    return provenance, exclusions


def _source_manifest_provenance(
    path: Path | None,
    *,
    fasta: Path,
    accession_count: int,
) -> dict[str, object] | None:
    if path is None:
        return None
    _require_file(path, "Protein sequence source manifest")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid protein sequence source manifest JSON: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != (
        SOURCE_MANIFEST_SCHEMA_VERSION
    ):
        raise SystemExit(f"Unsupported protein sequence source manifest schema: {path}")
    source = payload.get("source")
    artifact = payload.get("sequence_artifact")
    if not isinstance(source, dict) or not isinstance(artifact, dict):
        raise SystemExit("Protein sequence source manifest lacks source/sequence_artifact")
    raw_path = str(artifact.get("path") or "").strip()
    expected_sha = str(artifact.get("sha256") or "").strip()
    if not raw_path or len(expected_sha) != 64:
        raise SystemExit("Protein sequence source manifest lacks FASTA path/hash")
    artifact_path = Path(raw_path)
    if not artifact_path.is_absolute():
        artifact_path = path.parent / artifact_path
    if artifact_path.resolve() != fasta.resolve():
        raise SystemExit("Protein sequence source manifest is bound to a different FASTA")
    if _sha256(fasta) != expected_sha:
        raise SystemExit("Protein sequence source FASTA sha256 mismatch")
    try:
        expected_count = int(artifact.get("accession_count", -1))
    except (TypeError, ValueError) as exc:
        raise SystemExit("Protein sequence source accession count is invalid") from exc
    if expected_count != accession_count:
        raise SystemExit("Protein sequence source accession count mismatch")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "schema_version": SOURCE_MANIFEST_SCHEMA_VERSION,
        "source": source,
        "sequence_artifact": {
            "path": str(fasta.resolve()),
            "sha256": expected_sha,
            "accession_count": expected_count,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", required=True, type=Path)
    parser.add_argument(
        "--additional-fasta",
        action="append",
        default=[],
        type=Path,
        help="Additional protein FASTA to merge with --fasta; repeatable.",
    )
    parser.add_argument(
        "--sequence-universe-manifest",
        type=Path,
        help="Manifest binding evidence-derived supplemental FASTA and audited exclusions.",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        help="Manifest binding a source/release/license identity to --fasta.",
    )
    parser.add_argument("--cluster30-tsv", required=True, type=Path)
    parser.add_argument("--cluster50-tsv", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-manifest", required=True, type=Path)
    parser.add_argument("--mmseqs-version", required=True)
    parser.add_argument("--min-sequence-length", type=int, default=30)
    parser.add_argument("--coverage", type=float, default=0.80)
    parser.add_argument("--coverage-mode", type=int, default=0)
    parser.add_argument("--sensitivity", type=float, default=7.5)
    parser.add_argument("--max-seqs", type=int, default=1000)
    args = parser.parse_args()

    for output in (args.out_csv, args.out_manifest):
        output.unlink(missing_ok=True)
        output.with_suffix(output.suffix + ".tmp").unlink(missing_ok=True)
    if not 0.0 < args.coverage <= 1.0:
        raise SystemExit(f"--coverage must be in (0, 1]: {args.coverage}")
    if args.coverage_mode not in {0, 1, 2, 3, 4, 5}:
        raise SystemExit(f"--coverage-mode must be in [0, 5]: {args.coverage_mode}")
    if args.sensitivity <= 0.0:
        raise SystemExit(f"--sensitivity must be positive: {args.sensitivity}")
    if args.max_seqs < 1:
        raise SystemExit(f"--max-seqs must be >= 1: {args.max_seqs}")
    if args.min_sequence_length < 1:
        raise SystemExit(
            f"--min-sequence-length must be >= 1: {args.min_sequence_length}"
        )
    if not args.mmseqs_version.strip():
        raise SystemExit("--mmseqs-version must be nonblank")

    fasta_paths = [("base", args.fasta)] + [
        ("additional", path) for path in args.additional_fasta
    ]
    accessions, fasta_provenance = _merged_fasta_accessions(
        fasta_paths,
        min_sequence_length=args.min_sequence_length,
    )
    fasta_ids = set(accessions)
    source_manifest = _source_manifest_provenance(
        args.source_manifest,
        fasta=args.fasta,
        accession_count=len(accessions),
    )
    sequence_universe, excluded_targets = _sequence_universe_provenance(
        args.sequence_universe_manifest,
        base_fasta=args.fasta,
        additional_fastas=args.additional_fasta,
        fasta_ids=fasta_ids,
        min_sequence_length=args.min_sequence_length,
    )
    cluster30 = _cluster_assignments(
        args.cluster30_tsv,
        label="MMseqs2 30% cluster TSV",
        fasta_ids=fasta_ids,
    )
    cluster50 = _cluster_assignments(
        args.cluster50_tsv,
        label="MMseqs2 50% cluster TSV",
        fasta_ids=fasta_ids,
    )
    rows = [
        {
            "uniprot": accession,
            "target_cluster_30": cluster30[accession],
            "target_cluster_50": cluster50[accession],
        }
        for accession in sorted(accessions)
    ]
    _write_csv_atomic(rows, args.out_csv)
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "parameters": {
            "engine": "MMseqs2 easy-cluster",
            "mmseqs_version": args.mmseqs_version.strip(),
            "cluster_mode": "connected-component",
            "single_step_clustering": True,
            "coverage": args.coverage,
            "coverage_mode": args.coverage_mode,
            "sensitivity": args.sensitivity,
            "max_seqs": args.max_seqs,
            "min_sequence_length": args.min_sequence_length,
            "sequence_identity_thresholds": {"main": 0.30, "sensitivity": 0.50},
        },
        "inputs": {
            "fasta": str(args.fasta.resolve()),
            "fasta_sha256": _sha256(args.fasta),
            "fasta_files": fasta_provenance,
            "cluster30_tsv": str(args.cluster30_tsv.resolve()),
            "cluster30_tsv_sha256": _sha256(args.cluster30_tsv),
            "cluster50_tsv": str(args.cluster50_tsv.resolve()),
            "cluster50_tsv_sha256": _sha256(args.cluster50_tsv),
            "sequence_universe_manifest": sequence_universe,
            "source_manifest": source_manifest,
        },
        "excluded_targets": {
            "count": len(excluded_targets),
            "records": excluded_targets,
            "policy": (
                "No synthetic sequence cluster is assigned; only exclusions validated by the "
                "evidence target sequence universe manifest may be omitted downstream."
            ),
        },
        "artifact": {
            "path": str(args.out_csv.resolve()),
            "sha256": _sha256(args.out_csv),
            "rows": len(rows),
            "cluster30_count": len(set(cluster30.values())),
            "cluster50_count": len(set(cluster50.values())),
        },
    }
    _write_json_atomic(payload, args.out_manifest)
    print(
        "[target-clusters] wrote "
        f"rows={len(rows)} c30={payload['artifact']['cluster30_count']} "
        f"c50={payload['artifact']['cluster50_count']}"
    )


if __name__ == "__main__":
    main()
