#!/usr/bin/env bash
# stage0_download_alphafold.sh
# Resumable download of the AlphaFold human proteome v4 tar.
# Usage: stage0_download_alphafold.sh <URL> <OUTPUT_DIR>
# Idempotent: reuse a validated extracted proteome when it is already present.

set -euo pipefail

URL="${1:-https://ftp.ebi.ac.uk/pub/databases/alphafold/v4/UP000005640_9606_HUMAN_v4.tar}"
OUTDIR="${2:-data/alphafold_human_v4}"
TARNAME="$(basename "${URL}")"
TARGET="${OUTDIR}/${TARNAME}"

mkdir -p "${OUTDIR}"

echo "[stage0] AlphaFold human proteome download → ${TARGET}"
echo "[stage0] URL: ${URL}"

# A previous bootstrap may have materialized the proteome without retaining a
# complete tarball.  The extracted files are the actual downstream input, so
# avoid redownloading a multi-gigabyte archive when the extraction is valid.
MIN_EXISTING_MODELS="${MIN_EXISTING_MODELS:-20000}"
EXTRACTED_MARKER="${OUTDIR}/.extracted"
if [ -f "${EXTRACTED_MARKER}" ]; then
    EXISTING_PDB_MODELS=$(find "${OUTDIR}" -maxdepth 1 -type f \
        -name 'AF-*-F1-model_v4.pdb' -printf '.' | wc -c)
    EXISTING_CIF_MODELS=$(find "${OUTDIR}" -maxdepth 1 -type f \
        -name 'AF-*-F1-model_v4.cif.gz' -printf '.' | wc -c)
    if [ "${EXISTING_PDB_MODELS}" -ge "${MIN_EXISTING_MODELS}" ] \
       && [ "${EXISTING_CIF_MODELS}" -ge "${MIN_EXISTING_MODELS}" ]; then
        echo "[stage0] Reusing extracted AlphaFold proteome"\
             "(${EXISTING_PDB_MODELS} PDB and ${EXISTING_CIF_MODELS} canonical mmCIF files)."
        exit 0
    fi
    echo "[stage0] Existing extraction is incomplete"\
         "(${EXISTING_PDB_MODELS} PDB, ${EXISTING_CIF_MODELS} canonical mmCIF); "\
         "recovering from the pinned tar."
    rm -f "${EXTRACTED_MARKER}"
fi

# Disk pre-check: require ≥ 80 GB free on the target filesystem.
AVAIL_KB=$(df -k "${OUTDIR}" | awk 'NR==2 {print $4}')
AVAIL_GB=$(( AVAIL_KB / 1024 / 1024 ))
echo "[stage0] Free space on target FS: ${AVAIL_GB} GB"
if [ "${AVAIL_GB}" -lt 80 ]; then
    echo "[stage0][FATAL] Need ≥ 80 GB free, have ${AVAIL_GB} GB." >&2
    exit 2
fi

# Skip if already large enough (resumable safety net).
# v4 tar is ~4.8 GB compressed. Lower bound at 4 GB to detect partial downloads.
MIN_TAR_GB="${MIN_TAR_GB:-4}"
if [ -f "${TARGET}" ]; then
    SIZE_GB=$(du -sBG "${TARGET}" | awk '{print $1}' | tr -d 'G')
    if [ "${SIZE_GB}" -ge "${MIN_TAR_GB}" ]; then
        echo "[stage0] Tar present (${SIZE_GB} GB ≥ ${MIN_TAR_GB} GB) — skipping download."
        exit 0
    fi
fi

# wget -c for resume, -nv to keep logs reasonable.
wget -c -nv --tries=10 --waitretry=30 --read-timeout=60 \
     -O "${TARGET}" "${URL}"

FINAL_GB=$(du -sBG "${TARGET}" | awk '{print $1}' | tr -d 'G')
echo "[stage0] Download complete — ${TARGET} (${FINAL_GB} GB)."
