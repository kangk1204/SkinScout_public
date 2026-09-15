#!/usr/bin/env bash
# monitor_stage0.sh — append a timestamped Stage 0 progress line every INTERVAL
# seconds until the run is verified ready or the supplied Snakemake pid dies.
# Designed to be tailed: tail -f results/logs/stage0_progress.log
# Usage: monitor_stage0.sh <SNAKEMAKE_PID> [INTERVAL_SEC]

set -uo pipefail

usage() {
    echo "Usage: $0 <SNAKEMAKE_PID> [INTERVAL_SEC]" >&2
    exit 2
}

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    usage
fi

SMK_PID="$1"
INTERVAL="${2:-600}"
if ! [[ "${SMK_PID}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: SNAKEMAKE_PID must be a positive integer" >&2
    exit 2
fi
if ! [[ "${INTERVAL}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: INTERVAL_SEC must be a positive integer" >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# Gate manifests can contain repository-relative artifact paths.
cd -- "${ROOT}" || exit 1
LOG="${ROOT}/results/logs/stage0_progress.log"
FLAG="${ROOT}/data/manifests/stage0_complete.flag"
OPERATIONAL_GATE="${ROOT}/data/manifests/activity_retrieval_operational_gate.flag"
PYTHON_BIN="${PYTHON:-python3}"
TOTAL_PROT=20400

mkdir -p "$(dirname "${LOG}")"
echo "# Stage 0 progress monitor — started $(date -u +%FT%TZ), pid=${SMK_PID}, interval=${INTERVAL}s" >> "${LOG}"

while true; do
    ts="$(date -u +%FT%TZ)"
    clean=$(find "${ROOT}/data/human_clean" -name '*_clean.pdb' 2>/dev/null | wc -l)
    pockets=$(find "${ROOT}/data/human_pockets" -name '*.pockets.json' 2>/dev/null | wc -l)
    pdbqt=$(find "${ROOT}/data/human_pdbqt" -name '*.pdbqt' 2>/dev/null | wc -l)
    pct_line=$(grep -oE "[0-9]+ of [0-9]+ steps \([0-9]+%\) done" "${ROOT}/results/logs/stage0_run.log" 2>/dev/null | tail -1)

    # Do not infer this run's active rule from unrelated system-wide processes.
    stage="(running/scheduling)"
    p2rank_pct=""
    if [ "${pockets}" -gt 0 ]; then
        p2rank_pct=$(awk "BEGIN{printf \"%.1f%%\", ${pockets}*100/${TOTAL_PROT}}")
    fi

    echo "${ts} | step=${stage} | clean=${clean}/${TOTAL_PROT} | pockets=${pockets} ${p2rank_pct} | pdbqt=${pdbqt} | smk='${pct_line}'" >> "${LOG}"

    # Artifact presence is not readiness. Both strict validators must pass before
    # this monitor records a verified completion.
    if [ -f "${FLAG}" ] && [ -f "${OPERATIONAL_GATE}" ]; then
        if "${PYTHON_BIN}" "${ROOT}/scripts/stage0_verify.py" --repo "${ROOT}" \
                --strict --claim-quality --require-activity-evidence >/dev/null 2>&1 \
            && "${PYTHON_BIN}" "${ROOT}/scripts/validate_activity_retrieval_gate.py" \
                check-operational --gate "${OPERATIONAL_GATE}" >/dev/null 2>&1; then
            echo "${ts} | DONE — Stage 0 and operational readiness verified." >> "${LOG}"
            break
        fi
        echo "${ts} | readiness artifacts present, but strict verification failed; monitoring continues." >> "${LOG}"
    elif [ -f "${FLAG}" ]; then
        echo "${ts} | stage0_complete.flag present, but operational gate is missing; readiness not verified." >> "${LOG}"
    fi

    if ! kill -0 "${SMK_PID}" 2>/dev/null; then
        echo "${ts} | snakemake pid ${SMK_PID} exited (run finished or failed — check stage0_run.log); readiness not verified." >> "${LOG}"
        break
    fi
    sleep "${INTERVAL}"
done
