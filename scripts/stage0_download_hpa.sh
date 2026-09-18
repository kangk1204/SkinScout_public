#!/usr/bin/env bash
# stage0_download_hpa.sh
# Fetch Human Protein Atlas TSVs used for the SkinScore composite.
# Usage: stage0_download_hpa.sh <OUTDIR>
#
# URL layout (verified 2026-05): the consolidated `proteinatlas.tsv.zip` lives
# under /download/, while the per-assay tables live under /download/tsv/.
# `normal_tissue.tsv.zip` was retired by HPA and is not used by skin_score.

set -euo pipefail

OUTDIR="${1:?outdir required}"
mkdir -p "${OUTDIR}"
cd "${OUTDIR}"

# name|url  (pipe-separated). REQUIRED files fail the rule; OPTIONAL files
# only warn on 404 so a single retired table does not abort Stage 0.
REQUIRED=(
    "proteinatlas.tsv.zip|https://www.proteinatlas.org/download/proteinatlas.tsv.zip"
    "rna_tissue_consensus.tsv.zip|https://www.proteinatlas.org/download/tsv/rna_tissue_consensus.tsv.zip"
    "rna_single_cell_type.tsv.zip|https://www.proteinatlas.org/download/tsv/rna_single_cell_type.tsv.zip"
)

fetch() {
    local f="$1" url="$2" required="$3"
    if [ ! -f "${f}" ]; then
        if ! wget -c -nv "${url}" -O "${f}"; then
            rm -f "${f}"
            if [ "${required}" = "required" ]; then
                echo "[stage0.hpa][FATAL] required ${f} download failed: ${url}" >&2
                return 1
            fi
            echo "[stage0.hpa][WARN] optional ${f} unavailable (${url}) — skipping." >&2
            return 0
        fi
    fi
    if [ -f "${f}" ] && [ ! -f "${f%.zip}" ]; then
        unzip -o "${f}" -d .
    fi
    return 0
}

for entry in "${REQUIRED[@]}"; do
    name="${entry%%|*}"; url="${entry##*|}"
    fetch "${name}" "${url}" required
done

echo "[stage0.hpa] HPA tables ready in ${OUTDIR}"
