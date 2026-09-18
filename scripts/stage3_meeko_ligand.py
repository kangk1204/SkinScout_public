#!/usr/bin/env python3
"""stage3_meeko_ligand.py — SDF → ligand PDBQT for AutoDock-GPU."""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

LOG = logging.getLogger("stage3.meeko_ligand")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-sdf", required=True, type=Path)
    parser.add_argument("--out-pdbqt", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    args.out_pdbqt.parent.mkdir(parents=True, exist_ok=True)
    if not shutil.which("mk_prepare_ligand.py"):
        raise SystemExit("mk_prepare_ligand.py not on PATH (env: autodock_gpu)")
    cmd = [
        "mk_prepare_ligand.py",
        "-i", str(args.in_sdf),
        "-o", str(args.out_pdbqt),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        LOG.error("meeko failed: %s", res.stderr[-400:])
        sys.exit(res.returncode)
    LOG.info("Ligand PDBQT → %s", args.out_pdbqt)


if __name__ == "__main__":
    main()
