#!/usr/bin/env bash

# Resolve a Python runtime for scripts that may be called outside an activated
# SkinScout environment. The selected command is stored in SKINSCOUT_PYTHON_CMD.
skinscout_resolve_python() {
    local probe="${1:-pass}"
    local explicit="${PYTHON_BIN:-${PYTHON:-}}"
    local candidate
    local root
    local runner

    SKINSCOUT_PYTHON_CMD=()
    if [ -n "${explicit}" ]; then
        if ! command -v "${explicit}" >/dev/null 2>&1 && [ ! -x "${explicit}" ]; then
            echo "[skinscout.python][FATAL] configured Python is not executable: ${explicit}" >&2
            return 127
        fi
        if ! "${explicit}" -c "${probe}" >/dev/null 2>&1; then
            echo "[skinscout.python][FATAL] configured Python lacks required modules: ${explicit}" >&2
            return 5
        fi
        SKINSCOUT_PYTHON_CMD=("${explicit}")
        return 0
    fi

    for candidate in python python3; do
        if command -v "${candidate}" >/dev/null 2>&1 \
            && "${candidate}" -c "${probe}" >/dev/null 2>&1; then
            SKINSCOUT_PYTHON_CMD=("${candidate}")
            return 0
        fi
    done

    for root in \
        "${MAMBA_ROOT_PREFIX:-}" \
        "${HOME}/.local/share/micromamba" \
        "${HOME}/.local/share/mamba"; do
        candidate="${root}/envs/cosmax-base/bin/python"
        if [ -n "${root}" ] && [ -x "${candidate}" ] \
            && "${candidate}" -c "${probe}" >/dev/null 2>&1; then
            SKINSCOUT_PYTHON_CMD=("${candidate}")
            return 0
        fi
    done

    for runner in micromamba mamba; do
        if command -v "${runner}" >/dev/null 2>&1 \
            && "${runner}" run -n cosmax-base python -c "${probe}" >/dev/null 2>&1; then
            SKINSCOUT_PYTHON_CMD=("${runner}" run -n cosmax-base python)
            return 0
        fi
    done

    echo "[skinscout.python][FATAL] no Python runtime with the required modules was found." >&2
    echo "[skinscout.python][FATAL] Run scripts/bootstrap_runtime.sh or set PYTHON_BIN." >&2
    return 127
}
