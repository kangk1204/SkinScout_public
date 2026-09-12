#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${MAMBA_ROOT_PREFIX:-}" ]]; then
    for candidate in "${HOME}/.local/share/micromamba" "${HOME}/.local/share/mamba"; do
        if [[ -d "${candidate}/envs/cosmax-base" ]]; then
            MAMBA_ROOT_PREFIX="${candidate}"
            break
        fi
    done
fi
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/.local/share/micromamba}"
export MAMBA_ROOT_PREFIX

find_micromamba() {
    if [[ -n "${SKINSCOUT_MICROMAMBA:-}" && -x "${SKINSCOUT_MICROMAMBA}" ]]; then
        printf '%s\n' "${SKINSCOUT_MICROMAMBA}"
        return 0
    fi
    if command -v micromamba >/dev/null 2>&1; then
        command -v micromamba
        return 0
    fi
    if [[ -x "${HOME}/.local/bin/micromamba" ]]; then
        printf '%s\n' "${HOME}/.local/bin/micromamba"
        return 0
    fi
    printf '[SkinScout][ERROR] micromamba를 찾지 못했습니다. install_skinscout.sh를 다시 실행하세요.\n' >&2
    return 2
}

MAMBA="$(find_micromamba)"
action="start"
if (($#)) && [[ "$1" != -* ]]; then
    action="$1"
    shift
fi

case "${action}" in
    start)
        exec "${MAMBA}" run -n cosmax-base python \
            "${ROOT}/scripts/start_workbench.py" "$@"
        ;;
    run)
        exec "${MAMBA}" run -n cosmax-base python \
            "${ROOT}/scripts/run_skinscout.py" "$@"
        ;;
    doctor)
        "${MAMBA}" run -n cosmax-base python -c \
            'import pandas, rdkit, snakemake; from workbench import server' \
            >/dev/null
        printf '[SkinScout] 기본 conda 환경과 Workbench import 검증을 통과했습니다.\n'
        ;;
    status)
        pid_file="${ROOT}/results/logs/workbench.pid"
        if [[ -r "${pid_file}" ]]; then
            pid="$(<"${pid_file}")"
            if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
                printf '[SkinScout] Workbench가 실행 중입니다 (PID %s).\n' "${pid}"
                exit 0
            fi
        fi
        printf '[SkinScout] Workbench가 실행 중이 아닙니다.\n'
        exit 1
        ;;
    stop)
        pid_file="${ROOT}/results/logs/workbench.pid"
        if [[ ! -r "${pid_file}" ]]; then
            printf '[SkinScout] 실행 중인 Workbench가 없습니다.\n'
            exit 0
        fi
        pid="$(<"${pid_file}")"
        if [[ ! "${pid}" =~ ^[0-9]+$ ]] || ! kill -0 "${pid}" 2>/dev/null; then
            rm -f "${pid_file}"
            printf '[SkinScout] 실행 중인 Workbench가 없습니다.\n'
            exit 0
        fi
        command_line="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
        if [[ "${command_line}" != *"scripts/run_workbench.py"* ]]; then
            printf '[SkinScout][ERROR] PID 파일이 SkinScout Workbench를 가리키지 않습니다.\n' >&2
            exit 2
        fi
        kill "${pid}"
        rm -f "${pid_file}"
        printf '[SkinScout] Workbench를 종료했습니다.\n'
        ;;
    update|repair|rollback)
        printf '[SkinScout][ERROR] %s 명령은 서명된 Compose 배포에서만 사용할 수 있습니다.\n' \
            "${action}" >&2
        exit 2
        ;;
    *)
        printf '[SkinScout][ERROR] 알 수 없는 명령: %s\n' "${action}" >&2
        printf '사용법: skinscout [start|stop|status|doctor|run] [옵션]\n' >&2
        exit 2
        ;;
esac
