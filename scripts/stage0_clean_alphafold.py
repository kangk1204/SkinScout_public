#!/usr/bin/env python3
"""stage0_clean_alphafold.py

pLDDT-aware cleanup pass over the AlphaFold human proteome (v4).

For each AF-*-model_v4.pdb file:
  1. Parse pLDDT from the B-factor column.
  2. Remove residues with pLDDT < remove_cutoff (default 50).
  3. Identify a contiguous run of pLDDT < softmask_cutoff (default 70) at the
     N- or C-terminus; if run length ≥ run_length AND total trim length
     ≥ min_trim, drop the run.
  4. Write the trimmed PDB to <out_dir>/<UniProt>_clean.pdb.
  5. Write a soft-mask manifest <UniProt>_softmask.json listing residues whose
     pLDDT remains in [remove_cutoff, softmask_cutoff).

Implements INSTRUCTIONS.md §3.2(b).
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

LOG = logging.getLogger("stage0.clean")

UNIPROT_RE = re.compile(r"AF-(?P<uid>[A-Z0-9]+)-F(?P<fragment>\d+)-model_v\d+\.pdb")


@dataclass(frozen=True)
class CleanConfig:
    remove_cutoff: float
    softmask_cutoff: float
    run_length: int
    min_trim: int


@dataclass
class ResidueRecord:
    resnum: int
    plddt: float
    lines: list[str]


def parse_residues(pdb_path: Path) -> list[ResidueRecord]:
    by_res: dict[int, ResidueRecord] = {}
    with pdb_path.open("r") as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            try:
                resnum = int(line[22:26])
                bfac = float(line[60:66])
            except ValueError:
                continue
            rec = by_res.get(resnum)
            if rec is None:
                rec = ResidueRecord(resnum=resnum, plddt=bfac, lines=[])
                by_res[resnum] = rec
            rec.lines.append(line)
    return [by_res[k] for k in sorted(by_res)]


def decide_trim(residues: list[ResidueRecord], cfg: CleanConfig) -> tuple[int, int]:
    """Return (start_idx_keep, end_idx_keep) — half-open style indices."""
    n = len(residues)
    if n == 0:
        return 0, 0

    def low_run_from(idx: int, step: int) -> int:
        count = 0
        i = idx
        while 0 <= i < n and residues[i].plddt < cfg.softmask_cutoff:
            count += 1
            i += step
        return count

    n_run = low_run_from(0, 1)
    c_run = low_run_from(n - 1, -1)

    start = 0
    end = n
    if n_run >= cfg.run_length and n_run >= cfg.min_trim:
        start = n_run
    if c_run >= cfg.run_length and c_run >= cfg.min_trim:
        end = n - c_run
    if start >= end:
        # Pathological — keep at least the central residue.
        mid = n // 2
        start, end = mid, mid + 1
    return start, end


def write_cleaned(
    residues: list[ResidueRecord],
    out_pdb: Path,
    out_mask: Path,
    cfg: CleanConfig,
) -> None:
    soft_masked: list[int] = []
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    tmp_pdb = out_pdb.with_name(f".{out_pdb.name}.tmp")
    tmp_mask = out_mask.with_name(f".{out_mask.name}.tmp")
    with tmp_pdb.open("w") as fh:
        for rec in residues:
            if rec.plddt < cfg.remove_cutoff:
                continue
            if cfg.remove_cutoff <= rec.plddt < cfg.softmask_cutoff:
                soft_masked.append(rec.resnum)
            for line in rec.lines:
                fh.write(line)
        fh.write("END\n")
    tmp_mask.write_text(
        json.dumps(
            {
                "softmasked_residues": soft_masked,
                "remove_cutoff": cfg.remove_cutoff,
                "softmask_cutoff": cfg.softmask_cutoff,
            }
        )
    )
    tmp_pdb.replace(out_pdb)
    tmp_mask.replace(out_mask)


def remove_outputs(*paths: Path) -> None:
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def clean_one(args: tuple[Path, Path, CleanConfig]) -> tuple[str, bool, str]:
    src, out_dir, cfg = args
    m = UNIPROT_RE.match(src.name)
    if not m:
        return (src.name, False, "name pattern mismatch")
    uid = m.group(1)
    try:
        residues = parse_residues(src)
        if not residues:
            return (uid, False, "no parseable residues")
        start, end = decide_trim(residues, cfg)
        kept = [rec for rec in residues[start:end] if rec.plddt >= cfg.remove_cutoff]
        if not kept:
            return (uid, False, "no residues remain after pLDDT filtering")
        write_cleaned(
            kept,
            out_dir / f"{uid}_clean.pdb",
            out_dir / f"{uid}_softmask.json",
            cfg,
        )
        return (uid, True, "")
    except Exception as exc:  # noqa: BLE001 — log and continue
        return (uid, False, repr(exc))


def iter_inputs(af_dir: Path) -> Iterable[Path]:
    selected: dict[str, tuple[int, Path]] = {}
    for path in sorted(af_dir.rglob("AF-*-model_v*.pdb")):
        m = UNIPROT_RE.match(path.name)
        if not m:
            continue
        uid = m.group("uid")
        fragment = int(m.group("fragment"))
        current = selected.get(uid)
        if current is None or fragment < current[0]:
            selected[uid] = (fragment, path)
    for _uid, (_fragment, path) in sorted(selected.items()):
        yield path


def input_uid(src: Path) -> str | None:
    m = UNIPROT_RE.match(src.name)
    if not m:
        return None
    return m.group("uid")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--af-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--remove-cutoff", type=float, default=50.0)
    parser.add_argument("--softmask-cutoff", type=float, default=70.0)
    parser.add_argument("--run-length", type=int, default=5)
    parser.add_argument("--min-trim", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--min-success-count",
        type=int,
        default=None,
        help="Minimum unique cleaned receptors; defaults to all discovered inputs.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    cfg = CleanConfig(
        remove_cutoff=args.remove_cutoff,
        softmask_cutoff=args.softmask_cutoff,
        run_length=args.run_length,
        min_trim=args.min_trim,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    inputs = list(iter_inputs(args.af_dir))
    LOG.info("Cleaning %d AlphaFold PDBs → %s", len(inputs), args.out_dir)
    if not inputs:
        raise SystemExit("No AlphaFold PDB inputs were found")
    min_success_count = len(inputs) if args.min_success_count is None else args.min_success_count
    if min_success_count < 1:
        raise SystemExit("--min-success-count must be >= 1")

    for src in inputs:
        uid = input_uid(src)
        if uid is not None:
            remove_outputs(
                args.out_dir / f"{uid}_clean.pdb",
                args.out_dir / f"{uid}_softmask.json",
            )

    with tempfile.TemporaryDirectory(prefix=".stage0_clean_", dir=args.out_dir) as tmp:
        stage_dir = Path(tmp)
        work = [(p, stage_dir, cfg) for p in inputs]
        n_ok = n_fail = 0
        with mp.Pool(args.workers) as pool:
            for uid, ok, err in pool.imap_unordered(clean_one, work, chunksize=16):
                if ok:
                    n_ok += 1
                else:
                    n_fail += 1
                    LOG.warning("FAIL %s: %s", uid, err)
                if (n_ok + n_fail) % 500 == 0:
                    LOG.info("Progress: %d ok, %d fail", n_ok, n_fail)
        clean_outputs = list(stage_dir.glob("*_clean.pdb"))
        n_unique_ok = len(clean_outputs)
        LOG.info(
            "Done: %d ok, %d fail; %d unique cleaned receptors",
            n_ok,
            n_fail,
            n_unique_ok,
        )
        if n_unique_ok < min_success_count:
            raise SystemExit(
                "AlphaFold cleaning did not meet quality gate: "
                f"{n_unique_ok}/{len(inputs)} unique cleaned receptors "
                f"succeeded; required at least {min_success_count}"
            )
        for staged in sorted(stage_dir.iterdir()):
            staged.replace(args.out_dir / staged.name)


if __name__ == "__main__":
    main()
