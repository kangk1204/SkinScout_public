#!/usr/bin/env bash
# stage0_p2rank_batch.sh
# Batch pocket detection over every cleaned AlphaFold PDB.
# Usage: stage0_p2rank_batch.sh <CLEAN_DIR> <POCKET_DIR> <THREADS>

set -euo pipefail

CLEAN_DIR="${1:?clean dir required}"
POCKET_DIR="${2:?pocket dir required}"
THREADS="${3:-16}"

mkdir -p "${POCKET_DIR}"

# P2Rank batch mode = a dataset (.ds) file passed as the POSITIONAL arg, with
# one structure path per line. Paths must be absolute (P2Rank resolves relative
# paths against the .ds file's own directory, not the CWD). The `-f @list`
# form is NOT valid P2Rank syntax.
DS_FILE="${POCKET_DIR}/proteins.ds"
CLEAN_ABS="$(cd "${CLEAN_DIR}" && pwd)"
find "${CLEAN_ABS}" -maxdepth 1 -name "*_clean.pdb" | sort > "${DS_FILE}"

N=$(wc -l < "${DS_FILE}")
echo "[stage0.p2rank] ${N} cleaned PDBs → ${POCKET_DIR}"
if [ "${N}" -eq 0 ]; then
    echo "[stage0.p2rank][FATAL] no cleaned PDB inputs found in ${CLEAN_DIR}" >&2
    exit 2
fi

# Keep reruns fail-closed: stale CSVs or postprocessed manifests from a larger
# previous receptor set would otherwise be indistinguishable from fresh outputs.
# P2Rank can write dataset-specific nested directories, while postprocess scans
# recursively, so cleanup must cover the same tree.
find "${POCKET_DIR}" \
    -type f \
    \( -name "*_clean.pdb_predictions.csv" -o -name "*.pockets.json" \) \
    -delete

if ! command -v prank >/dev/null 2>&1; then
    echo "[stage0.p2rank][FATAL] 'prank' (P2Rank) not on PATH." >&2
    exit 3
fi

prank predict "${DS_FILE}" \
      -threads "${THREADS}" \
      -o "${POCKET_DIR}" \
      -c alphafold

echo "[stage0.p2rank] Done."
