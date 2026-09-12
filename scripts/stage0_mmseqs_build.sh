#!/usr/bin/env bash
# stage0_mmseqs_build.sh
# Build two MMseqs2 databases used for evaluation leakage audits:
#   * human_db: canonical UniProt human reference proteome
#   * training_cutoff_db: PDB sequences deposited ≤ 2021-09-30 (Boltz-2 cutoff)
# Usage: stage0_mmseqs_build.sh <CANONICAL_FASTA> <OUTDIR> [--allow-missing-training-cutoff]

set -euo pipefail

HUMAN_FA="${1:?canonical human FASTA required}"
OUTDIR="${2:?out dir required}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=python_runtime.sh
source "${SCRIPT_DIR}/python_runtime.sh"
ALLOW_MISSING_TRAINING_CUTOFF=0
if [ "${3:-}" = "--allow-missing-training-cutoff" ]; then
    ALLOW_MISSING_TRAINING_CUTOFF=1
elif [ "${3:-}" != "" ]; then
    echo "[stage0.mmseqs][FATAL] unknown option: ${3}" >&2
    exit 2
fi
mkdir -p "${OUTDIR}"
TMP="${OUTDIR}/tmp"
mkdir -p "${TMP}"
MMSEQS_THREADS="${MMSEQS_THREADS:-${SLURM_CPUS_PER_TASK:-$(nproc)}}"

if ! command -v mmseqs >/dev/null 2>&1; then
    echo "[stage0.mmseqs][FATAL] mmseqs not on PATH" >&2
    exit 5
fi

skinscout_resolve_python
if [ ! -s "${HUMAN_FA}" ]; then
    echo "[stage0.mmseqs][FATAL] canonical human FASTA missing or empty: ${HUMAN_FA}" >&2
    exit 4
fi

# --- 2) Build current-proteome MMseqs DB ---
HUMAN_DB="${OUTDIR}/human_db"
mmseqs createdb "${HUMAN_FA}" "${HUMAN_DB}"
mmseqs createindex "${HUMAN_DB}" "${TMP}" --search-type 2 --threads "${MMSEQS_THREADS}"

# --- 3) Training-cutoff DB (PDB seqs <= 2021-09-30) ---
TRAIN_FA="${OUTDIR}/training_cutoff_seqs.fasta"
if [ ! -s "${TRAIN_FA}" ] && [ "${ALLOW_MISSING_TRAINING_CUTOFF}" != "1" ]; then
    echo "[stage0.mmseqs] training_cutoff_seqs.fasta missing; building from public RCSB derived data." >&2
    "${SKINSCOUT_PYTHON_CMD[@]}" "${SCRIPT_DIR}/stage0_fetch_training_cutoff.py" \
        --out-dir "${OUTDIR}" \
        --cutoff-date "${RCSB_TRAINING_CUTOFF_DATE:-2021-09-30}" \
        --entries-url "${RCSB_ENTRIES_IDX_URL:-https://files.rcsb.org/pub/pdb/derived_data/index/entries.idx}" \
        --seqres-url "${RCSB_SEQRES_FASTA_URL:-https://files.rcsb.org/pub/pdb/derived_data/pdb_seqres.txt.gz}" \
        --min-sequences "${RCSB_MIN_TRAINING_CUTOFF_SEQUENCES:-10000}"
fi

if [ -f "${TRAIN_FA}" ]; then
    TRAIN_DB="${OUTDIR}/training_cutoff_db"
    mmseqs createdb "${TRAIN_FA}" "${TRAIN_DB}"
    mmseqs createindex "${TRAIN_DB}" "${TMP}" --search-type 2 --threads "${MMSEQS_THREADS}"
else
    if [ "${ALLOW_MISSING_TRAINING_CUTOFF}" = "1" ]; then
        echo "[stage0.mmseqs][WARN] training_cutoff_seqs.fasta missing; sequence leakage audit disabled for explicit diagnostics." >&2
    else
        echo "[stage0.mmseqs][FATAL] training_cutoff_seqs.fasta is required for claim-quality leakage audit: ${TRAIN_FA}" >&2
        echo "[stage0.mmseqs][FATAL] Use --allow-missing-training-cutoff only for explicit partial diagnostics." >&2
        exit 6
    fi
fi

echo "[stage0.mmseqs] done."
