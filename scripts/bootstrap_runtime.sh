#!/usr/bin/env bash
# Bootstrap the user-local SkinScout runtime on Ubuntu.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY_RUN=0
WITH_TARGET_MODELS=0
PROFILE="demo"
PROFILE_EXPLICIT=0
ANALOG_BUNDLE=""
ANALOG_BUNDLE_MANIFEST=""
ANALOG_BUNDLE_MANIFEST_SHA256=""
RUNTIME_MODE="conda"
RUNTIME_ROOT="${SKINSCOUT_RUNTIME_ROOT:-${HOME}/.local/share/skinscout}"
STATE_DIR="${SKINSCOUT_STATE_DIR:-${RUNTIME_ROOT}/state}"
SECRETS_DIR="${SKINSCOUT_SECRETS_DIR:-${RUNTIME_ROOT}/secrets}"

usage() {
    printf '%s\n' \
        "Usage: bash scripts/bootstrap_runtime.sh [--dry-run] [--runtime compose|conda] [--profile analog|demo|full] [--analog-bundle URL|PATH] [--analog-bundle-manifest URL|PATH] [--analog-bundle-manifest-sha256 HEX] [--with-target-models]" \
        "" \
        "Prepares the beginner conda runtime by default. The signed Compose" \
        "runtime remains opt-in with --runtime compose. --with-target-models is" \
        "a deprecated alias for --profile full."
}

while (($#)); do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --runtime) shift; RUNTIME_MODE="${1:?--runtime requires compose or conda}" ;;
        --profile) shift; PROFILE="${1:?--profile requires analog, demo or full}"; PROFILE_EXPLICIT=1 ;;
        --analog-bundle) shift; ANALOG_BUNDLE="${1:?--analog-bundle requires a URL or path}" ;;
        --analog-bundle-manifest) shift; ANALOG_BUNDLE_MANIFEST="${1:?--analog-bundle-manifest requires a URL or path}" ;;
        --analog-bundle-manifest-sha256) shift; ANALOG_BUNDLE_MANIFEST_SHA256="${1:?--analog-bundle-manifest-sha256 requires a 64-char hex digest}" ;;
        --with-target-models) WITH_TARGET_MODELS=1 ;;
        -h|--help) usage; exit 0 ;;
        *) printf '[SkinScout][ERROR] Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

if ((WITH_TARGET_MODELS && PROFILE_EXPLICIT)); then
    printf '[SkinScout][ERROR] --with-target-models is a deprecated alias for --profile full and cannot be combined with --profile\n' >&2
    exit 2
fi

if ((WITH_TARGET_MODELS)); then
    PROFILE="full"
fi

if [[ "${PROFILE}" != "analog" && "${PROFILE}" != "demo" && "${PROFILE}" != "full" ]]; then
    printf '[SkinScout][ERROR] --profile must be analog, demo or full\n' >&2
    exit 2
fi

# analog 는 대체소재 검색만 쓰는 설치다. Stage 0(125 GB)을 만들지 않고 만들어진
# 데이터 번들을 받으므로, 번들 위치가 없으면 시작할 이유가 없다 - 끝까지 깔고
# 나서 "데이터가 없습니다"를 보여 주는 것보다 여기서 멈추는 편이 낫다.
if [[ "${PROFILE}" == "analog" && -z "${ANALOG_BUNDLE}" ]]; then
    printf '[SkinScout][ERROR] --profile analog requires --analog-bundle <URL or path>\n' >&2
    exit 2
fi

if [[ "${RUNTIME_MODE}" != "compose" && "${RUNTIME_MODE}" != "conda" ]]; then
    printf '[SkinScout][ERROR] --runtime must be compose or conda\n' >&2
    exit 2
fi

log() {
    printf '[SkinScout] %s\n' "$*"
}

run() {
    if ((DRY_RUN)); then
        printf '[SkinScout][dry-run]'
        printf ' %q' "$@"
        printf '\n'
        return 0
    fi
    "$@"
}

if [[ ! -r /etc/os-release ]]; then
    printf '[SkinScout][ERROR] Ubuntu 정보를 확인할 수 없습니다.\n' >&2
    exit 2
fi

# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "24.04" || "$(uname -m)" != "x86_64" ]]; then
    if [[ "${RUNTIME_MODE}" == "conda" ]] && ((DRY_RUN == 0)); then
        printf '[SkinScout][ERROR] 이 conda 자동 설치는 Ubuntu 24.04 x86_64에서만 지원합니다.\n' >&2
        exit 2
    fi
    log "conda 실설치 지원 환경은 Ubuntu 24.04 x86_64입니다."
fi

missing_commands=()
for command in curl tar bzip2 python3; do
    command -v "${command}" >/dev/null 2>&1 || missing_commands+=("${command}")
done
if ((${#missing_commands[@]})); then
    if ((DRY_RUN)); then
        log "실제 설치 전에 install_skinscout.sh가 준비할 기본 도구: ${missing_commands[*]}"
    else
        printf '[SkinScout][ERROR] 기본 도구가 필요합니다: %s\n' "${missing_commands[*]}" >&2
        printf '[SkinScout][ERROR] 먼저 저장소의 install_skinscout.sh를 실행하세요.\n' >&2
        exit 2
    fi
fi

write_state() {
    local state="$1"
    local reboot_required="$2"
    if ((DRY_RUN)); then
        log "installer state 기록 예정: ${state}, reboot_required=${reboot_required}"
        return 0
    fi
    mkdir -p "${STATE_DIR}"
    python3 - "$STATE_DIR/bootstrap_state.json" "$state" "$reboot_required" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
path.write_text(
    json.dumps(
        {
            "state": sys.argv[2],
            "reboot_required": sys.argv[3] == "1",
            "resume_command": "bash scripts/bootstrap_runtime.sh --runtime compose",
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
}

secure_boot_enabled() {
    if command -v mokutil >/dev/null 2>&1; then
        mokutil --sb-state 2>/dev/null | grep -qi 'enabled'
        return $?
    fi
    return 1
}

ensure_compose_runtime() {
    missing_runtime=()
    command -v docker >/dev/null 2>&1 || missing_runtime+=(docker)
    if ! docker compose version >/dev/null 2>&1; then
        missing_runtime+=(docker-compose-plugin)
    fi
    command -v nvidia-smi >/dev/null 2>&1 || missing_runtime+=(nvidia-driver nvidia-container-toolkit)
    command -v cosign >/dev/null 2>&1 || missing_runtime+=(cosign)

    mkdir_msg="${RUNTIME_ROOT}/{state,data,results,cache,secrets}"
    if ((DRY_RUN)); then
        log "사용자 런타임 디렉터리 준비 예정: ${mkdir_msg}"
    else
        mkdir -p "${RUNTIME_ROOT}/state" \
            "${RUNTIME_ROOT}/data" \
            "${RUNTIME_ROOT}/results" \
            "${RUNTIME_ROOT}/cache" \
            "${SECRETS_DIR}"
        if [[ ! -f "${SECRETS_DIR}/worker-api.token" ]]; then
            python3 - <<'PY' > "${SECRETS_DIR}/worker-api.token"
import secrets
print(secrets.token_urlsafe(32))
PY
            chmod 0600 "${SECRETS_DIR}/worker-api.token"
        fi
    fi

    if ((${#missing_runtime[@]})); then
        log "관리자 단계 필요: ${missing_runtime[*]}"
        log "대용량 Stage 0/공개 데이터는 런타임 준비 후 별도로 활성화합니다."
        if secure_boot_enabled; then
            write_state "root_phase_reboot_required" 1
            log "Secure Boot가 켜져 있습니다. NVIDIA 모듈/MOK 등록 후 재부팅하고 설치를 다시 실행하세요."
            return 0
        fi
        write_state "root_phase_required" 0
        if ((DRY_RUN)); then
            log "Docker Engine, NVIDIA Container Toolkit, Cosign 설치 예정"
            return 0
        fi
        printf '[SkinScout][ERROR] Docker/NVIDIA/Cosign 관리자 설치가 필요합니다. install_skinscout.sh를 사용하세요.\n' >&2
        return 2
    fi

    if ((DRY_RUN)); then
        log "Compose 이미지 서명/다이제스트 및 GPU/NVML/CUDA 런타임 확인 예정"
    elif [[ ! -f "${ROOT}/scripts/host_cli.py" ]]; then
        write_state "readiness_blocked" 0
        printf '[SkinScout][ERROR] Compose 런타임 doctor를 찾을 수 없습니다.\n' >&2
        return 2
    else
        write_state "tools_installed" 0
        python3 "${ROOT}/scripts/host_cli.py" doctor >/dev/null || {
            write_state "readiness_blocked" 0
            printf '[SkinScout][ERROR] Compose 배포 정책 또는 GPU/NVML/CUDA 런타임 확인 실패\n' >&2
            return 2
        }
    fi
    write_state "ready" 0
    log "Compose 실행 환경 준비가 끝났습니다."
    log "대용량 Stage 0/공개 데이터는 skinscout repair --manifest <manifest.json> 또는 data_cas.py로 활성화하세요."
}

if [[ "${RUNTIME_MODE}" == "compose" ]]; then
    ensure_compose_runtime
    exit $?
fi

helper_args=(--runtime-root "${RUNTIME_ROOT}" --profile "${PROFILE}")
[[ -n "${ANALOG_BUNDLE}" ]] && helper_args+=(--analog-bundle "${ANALOG_BUNDLE}")
[[ -n "${ANALOG_BUNDLE_MANIFEST}" ]] && helper_args+=(--analog-bundle-manifest "${ANALOG_BUNDLE_MANIFEST}")
[[ -n "${ANALOG_BUNDLE_MANIFEST_SHA256}" ]] && helper_args+=(--analog-bundle-manifest-sha256 "${ANALOG_BUNDLE_MANIFEST_SHA256}")
((DRY_RUN)) && helper_args+=(--dry-run)
exec python3 "${ROOT}/scripts/install_runtime.py" "${helper_args[@]}" "$@"
