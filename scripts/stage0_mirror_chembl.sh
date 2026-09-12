#!/usr/bin/env bash
# stage0_mirror_chembl.sh
# Mirror ChEMBL SQLite + human single-protein activity evidence slice.
# Usage: stage0_mirror_chembl.sh <OUTDIR> [release] [release-date]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=python_runtime.sh
source "${SCRIPT_DIR}/python_runtime.sh"
skinscout_resolve_python "import pandas, pyarrow"

OUTDIR="${1:?outdir required}"
RELEASE="${2:-37}"
RELEASE_DATE="${3:-}"
mkdir -p "${OUTDIR}"

BASE="https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_${RELEASE}"
TGZ="chembl_${RELEASE}_sqlite.tar.gz"

if [ ! -f "${OUTDIR}/${TGZ}" ]; then
    echo "[stage0.chembl] downloading ${TGZ} (~5 GB)"
    wget -c -nv -O "${OUTDIR}/${TGZ}" "${BASE}/${TGZ}"
fi

DB_PATH="$(find "${OUTDIR}/chembl_${RELEASE}" -type f -name "chembl_${RELEASE}.db" -print -quit 2>/dev/null || true)"
if [ -z "${DB_PATH}" ]; then
    echo "[stage0.chembl] extracting"
    tar -xzf "${OUTDIR}/${TGZ}" -C "${OUTDIR}"
fi

DB_PATH="$(find "${OUTDIR}/chembl_${RELEASE}" -type f -name "chembl_${RELEASE}.db" -print -quit 2>/dev/null || true)"
if [ -z "${DB_PATH}" ]; then
    echo "[stage0.chembl][FATAL] chembl_${RELEASE}.db not found under ${OUTDIR}/chembl_${RELEASE}" >&2
    exit 4
fi

CHEMBL_ARGS=(
    --db "${DB_PATH}"
    --out-dir "${OUTDIR}"
    --release "${RELEASE}"
    --source-archive "${OUTDIR}/${TGZ}"
)
if [ -n "${RELEASE_DATE}" ]; then
    CHEMBL_ARGS+=(--release-date "${RELEASE_DATE}")
fi
"${SKINSCOUT_PYTHON_CMD[@]}" "${SCRIPT_DIR}/build_chembl_activity_evidence.py" \
    "${CHEMBL_ARGS[@]}"

echo "[stage0.chembl] done."
