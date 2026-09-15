#!/usr/bin/env python3
"""Ask whether PubChem BioAssay covers the targets this tool actually misses.

Priority 4 of `docs/DATA_EXPANSION_20260831.md` said not to adopt PubChem on the
strength of "it is large". Size is not the question - net increment is. PubChem
BioAssay is tens of GB, so rather than download it to find out, this asks PubChem
about the specific accessions our gap consists of.

For each accession it fetches the assay ids PubChem holds, then the concise
bioactivity table, and counts two different things: how many compounds appear at
all, and how many carry a numeric `Activity Value [uM]`. Only the second kind can
become a pActivity, so only the second kind would make a target rankable here.

It is polite by construction: PubChem asks for at most 5 requests/second, this
does far fewer, caches every answer to disk, and resumes rather than refetching.

Run:
    python scripts/measure_pubchem_increment.py \\
        --accessions data/curation/skin_target_worklist.csv \\
        --cache data/curation/pubchem_probe_cache.json
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/protein/accession"
USER_AGENT = "SkinScout-coverage-probe/1.0 (research; contact via repository)"
# PubChem's own guidance is <=5 requests/second. Stay well under it.
DELAY_SECONDS = 0.35
RETRIES = 3


def _get(url: str, timeout: float) -> str | None:
    """Return the body, None for 404 (no data), raising only on repeated failure."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            last = error
            # 503 means we are being asked to slow down; do exactly that.
            time.sleep(2.0 * (attempt + 1))
        except Exception as error:  # noqa: BLE001 - network, retried
            last = error
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"giving up on {url}: {last}")


def probe(accession: str, timeout: float) -> dict[str, object]:
    """Assay count, and how much of it is a usable potency.

    "PubChem has data" is not the same as "PubChem has data this tool could rank".
    A row with no `Activity Value [uM]` - a binding screen reported only as an
    outcome - cannot become a pActivity, so it is counted separately.
    """
    aids_body = _get(f"{BASE}/{accession}/aids/TXT", timeout)
    if aids_body is None:
        return {"assays": 0, "compounds": 0, "measured_compounds": 0, "measured_rows": 0}
    assays = [line for line in aids_body.splitlines() if line.strip()]
    time.sleep(DELAY_SECONDS)

    body = _get(f"{BASE}/{accession}/concise/CSV", timeout)
    if body is None:
        return {
            "assays": len(assays),
            "compounds": 0,
            "measured_compounds": 0,
            "measured_rows": 0,
        }

    import csv
    import io

    compounds: set[str] = set()
    measured: set[str] = set()
    measured_rows = 0
    reader = csv.DictReader(io.StringIO(body))
    for row in reader:
        cid = (row.get("CID") or "").strip()
        if not cid:
            continue
        compounds.add(cid)
        value = (row.get("Activity Value [uM]") or "").strip()
        if value:
            try:
                float(value)
            except ValueError:
                continue
            measured.add(cid)
            measured_rows += 1
    return {
        "assays": len(assays),
        "compounds": len(compounds),
        "measured_compounds": len(measured),
        "measured_rows": measured_rows,
    }


def _accessions(path: Path, column: str, only_worth: bool) -> list[str]:
    import pandas as pd

    frame = pd.read_csv(path)
    if only_worth and "worth_curating" in frame.columns:
        frame = frame[frame["worth_curating"]]
    return sorted(frame[column].dropna().astype(str).unique())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--accessions",
        type=Path,
        default=ROOT / "data" / "curation" / "skin_target_worklist.csv",
        help="CSV with a uniprot column",
    )
    parser.add_argument("--column", default="uniprot")
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="probe every row, not only the ones marked worth_curating",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "data" / "curation" / "pubchem_probe_cache.json",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--limit", type=int, help="probe at most N new accessions")
    args = parser.parse_args()

    accessions = _accessions(args.accessions, args.column, not args.all_rows)
    cache: dict[str, dict] = {}
    if args.cache.exists():
        cache = json.loads(args.cache.read_text(encoding="utf-8"))

    todo = [a for a in accessions if a not in cache]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(accessions)} accession(s); {len(cache)} cached, probing {len(todo)}")

    for index, accession in enumerate(todo, start=1):
        cache[accession] = probe(accession, args.timeout)
        if index % 10 == 0 or index == len(todo):
            args.cache.parent.mkdir(parents=True, exist_ok=True)
            args.cache.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"  {index}/{len(todo)}", flush=True)
        time.sleep(DELAY_SECONDS)

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    args.cache.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    probed = {a: cache[a] for a in accessions if a in cache}
    with_assays = {a: v for a, v in probed.items() if v["assays"] > 0}
    with_measured = {a: v for a, v in probed.items() if v.get("measured_compounds", 0) > 0}

    print()
    print(f"probed                        : {len(probed):>5,}")
    print(f"  PubChem has any assay       : {len(with_assays):>5,}")
    print(f"  ...with a measured potency  : {len(with_measured):>5,}   <- the usable ones")
    if with_measured:
        top = sorted(with_measured.items(), key=lambda kv: -kv[1]["measured_compounds"])[:20]
        print("\n  most measured compounds:")
        for accession, value in top:
            print(
                f"    {accession}  assays={value['assays']:>4}  "
                f"compounds={value['compounds']:>6,}  measured={value['measured_compounds']:>6,}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
