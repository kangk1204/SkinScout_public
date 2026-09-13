#!/usr/bin/env python3
"""Build the Stage 0 MMseqs training-cutoff FASTA from public RCSB data."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import shutil
import sys
import tempfile
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Iterator, TextIO
from urllib.parse import urlparse
from urllib.request import urlopen


LOG = logging.getLogger("stage0.mmseqs.training_cutoff")
DEFAULT_ENTRIES_URL = "https://files.rcsb.org/pub/pdb/derived_data/index/entries.idx"
DEFAULT_SEQRES_URL = "https://files.rcsb.org/pub/pdb/derived_data/pdb_seqres.txt.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cutoff-date", default="2021-09-30")
    parser.add_argument("--entries-url", default=DEFAULT_ENTRIES_URL)
    parser.add_argument("--seqres-url", default=DEFAULT_SEQRES_URL)
    parser.add_argument("--out-name", default="training_cutoff_seqs.fasta")
    parser.add_argument("--manifest-name", default="training_cutoff_manifest.json")
    parser.add_argument("--min-sequences", type=int, default=10000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_name(source: str, fallback: str) -> str:
    parsed = urlparse(source)
    if parsed.scheme and parsed.scheme != "file":
        return Path(parsed.path).name or fallback
    return Path(parsed.path or source).name or fallback


def fetch_source(source: str, dest: Path) -> None:
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        LOG.info("Downloading %s -> %s", source, dest)
        with urlopen(source, timeout=120) as response, dest.open("wb") as out:
            shutil.copyfileobj(response, out)
        return
    local = Path(parsed.path if parsed.scheme == "file" else source)
    if not local.exists():
        raise FileNotFoundError(f"source not found: {source}")
    LOG.info("Copying %s -> %s", local, dest)
    shutil.copy2(local, dest)


@contextmanager
def open_maybe_gzip(path: Path) -> Iterator[TextIO]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            yield fh
    else:
        with path.open("rt", encoding="utf-8", errors="replace") as fh:
            yield fh


def parse_idx_date(raw: str) -> date:
    raw = raw.strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            pass
    try:
        parsed = datetime.strptime(raw, "%m/%d/%y").date()
    except ValueError as exc:
        raise ValueError(f"unsupported RCSB date: {raw!r}") from exc
    if parsed.year > 2030:
        parsed = parsed.replace(year=parsed.year - 100)
    return parsed


def cutoff_entries(entries_idx: Path, cutoff: date) -> set[str]:
    allowed: set[str] = set()
    with entries_idx.open("rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith(("IDCODE,", "-------", "PROTEIN DATA BANK")):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            pdb_id = parts[0].strip().upper()
            if not pdb_id or pdb_id == "IDCODE":
                continue
            try:
                accession_date = parse_idx_date(parts[2])
            except ValueError:
                continue
            if accession_date <= cutoff:
                allowed.add(pdb_id)
    if not allowed:
        raise RuntimeError(f"no entries found at or before cutoff {cutoff.isoformat()}")
    return allowed


def fasta_record_entry_id(header: str) -> str:
    token = header[1:].split(None, 1)[0]
    return token.split("_", 1)[0].split(".", 1)[0].upper()


def write_cutoff_fasta(seqres: Path, allowed_entries: set[str], out_fasta: Path) -> int:
    count = 0
    keep = False
    wrote_sequence = False
    with open_maybe_gzip(seqres) as inp, out_fasta.open("wt", encoding="utf-8") as out:
        for raw in inp:
            line = raw.rstrip("\n")
            if line.startswith(">"):
                keep = (
                    "mol:protein" in line.lower()
                    and fasta_record_entry_id(line) in allowed_entries
                )
                if keep:
                    out.write(line + "\n")
                    count += 1
                    wrote_sequence = False
                continue
            if keep:
                seq = line.strip()
                if seq:
                    out.write(seq + "\n")
                    wrote_sequence = True
                elif wrote_sequence:
                    out.write("\n")
    return count


def count_fasta_records(path: Path) -> int:
    count = 0
    with open_maybe_gzip(path) as fh:
        for line in fh:
            if line.startswith(">"):
                count += 1
    return count


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    cutoff = datetime.strptime(args.cutoff_date, "%Y-%m-%d").date()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_fasta = args.out_dir / args.out_name
    manifest_path = args.out_dir / args.manifest_name
    if out_fasta.exists() and out_fasta.stat().st_size > 0 and not args.force:
        sequence_count = count_fasta_records(out_fasta)
        if sequence_count < args.min_sequences:
            raise RuntimeError(
                f"existing training cutoff FASTA has {sequence_count} records; "
                f"expected at least {args.min_sequences}"
            )
        if not manifest_path.exists():
            manifest_path.write_text(
                json.dumps(
                    {
                        "cutoff_date": cutoff.isoformat(),
                        "protein_sequence_count": sequence_count,
                        "out_fasta": str(out_fasta),
                        "out_fasta_sha256": sha256_file(out_fasta),
                        "reused_existing_fasta": True,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
        LOG.info("Using existing %s (%d records)", out_fasta, sequence_count)
        return 0

    with tempfile.TemporaryDirectory(prefix="stage0_rcsb_", dir=str(args.out_dir)) as tmp:
        tmp_dir = Path(tmp)
        entries_path = tmp_dir / source_name(args.entries_url, "entries.idx")
        seqres_path = tmp_dir / source_name(args.seqres_url, "pdb_seqres.txt.gz")
        fetch_source(args.entries_url, entries_path)
        fetch_source(args.seqres_url, seqres_path)
        allowed = cutoff_entries(entries_path, cutoff)
        tmp_fasta = tmp_dir / args.out_name
        sequence_count = write_cutoff_fasta(seqres_path, allowed, tmp_fasta)
        if sequence_count < args.min_sequences:
            raise RuntimeError(
                f"training cutoff FASTA has {sequence_count} protein sequences; "
                f"expected at least {args.min_sequences}"
            )
        tmp_fasta.replace(out_fasta)
        manifest = {
            "cutoff_date": cutoff.isoformat(),
            "entries_url": args.entries_url,
            "entries_sha256": sha256_file(entries_path),
            "seqres_url": args.seqres_url,
            "seqres_sha256": sha256_file(seqres_path),
            "cutoff_entry_count": len(allowed),
            "protein_sequence_count": sequence_count,
            "out_fasta": str(out_fasta),
            "out_fasta_sha256": sha256_file(out_fasta),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    LOG.info("Wrote %s protein sequences to %s", sequence_count, out_fasta)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"[stage0.mmseqs][FATAL] failed to build training cutoff FASTA: {exc}", file=sys.stderr)
        raise SystemExit(1)
