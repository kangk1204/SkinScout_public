#!/usr/bin/env python3
"""boltz_affinity_demo.py — Boltz-2 co-folding + affinity for one protein–ligand pair.

Runs in the `cosmax-boltz2` conda env from `envs/boltz2.yml`.
Builds the Boltz YAML, runs `boltz predict --no_kernels`, and parses the
affinity head into a one-line summary.

Boltz-2 affinity outputs (per affinity_<name>.json):
  affinity_pred_value         — predicted log-scale binding value (lower = stronger)
  affinity_probability_binary — P(binder) in [0,1] (higher = more likely a binder)

Usage:
  python boltz_affinity_demo.py --uniprot P14679 --smiles "OCC1=CC(=O)C(O)=CO1" \
      --name kojic_TYR --out-dir results/runs/boltz_tests
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from pathlib import Path

import requests

LOG = logging.getLogger("boltz.affinity")


def fetch_sequence(uniprot: str) -> str:
    r = requests.get(f"https://rest.uniprot.org/uniprotkb/{uniprot}.fasta", timeout=30)
    r.raise_for_status()
    return "".join(r.text.splitlines()[1:])


def write_yaml(seq: str, smiles: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "version: 1\n"
        "sequences:\n"
        f"  - protein: {{id: A, sequence: {seq}}}\n"
        f'  - ligand: {{id: B, smiles: "{smiles}"}}\n'
        "properties:\n"
        "  - affinity: {binder: B}\n"
    )


def parse_affinity(out_dir: Path) -> dict | None:
    for f in out_dir.rglob("affinity_*.json"):
        return json.loads(f.read_text())
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--uniprot", required=True)
    p.add_argument("--smiles", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--diffusion-samples", type=int, default=1)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not shutil.which("boltz"):
        raise SystemExit("boltz not on PATH (activate cosmax-boltz2)")

    seq = fetch_sequence(args.uniprot)
    LOG.info("%s: %s (%d aa) + ligand", args.name, args.uniprot, len(seq))
    yaml = args.out_dir / f"{args.name}.yaml"
    write_yaml(seq, args.smiles, yaml)
    run_out = args.out_dir / f"{args.name}_out"
    if run_out.exists():
        shutil.rmtree(run_out)

    cmd = ["boltz", "predict", str(yaml), "--out_dir", str(run_out),
           "--use_msa_server", "--accelerator", "gpu",
           "--diffusion_samples", str(args.diffusion_samples), "--no_kernels"]
    LOG.info("running: %s", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        LOG.error("boltz failed:\n%s", res.stderr[-1500:])
        raise SystemExit(res.returncode)

    aff = parse_affinity(run_out)
    if aff is None:
        raise SystemExit("no affinity json produced")
    LOG.info("%s → pred_value=%.3f  binder_prob=%.3f",
             args.name, aff["affinity_pred_value"], aff["affinity_probability_binary"])
    summary = run_out / "affinity_summary.json"
    summary.write_text(json.dumps({"name": args.name, "uniprot": args.uniprot,
                                   "smiles": args.smiles, **aff}, indent=2))
    print(json.dumps({"name": args.name, "uniprot": args.uniprot, **aff}))


if __name__ == "__main__":
    main()
