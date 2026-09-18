#!/usr/bin/env bash
# Install SkinScout and its beginner Workbench on Ubuntu.

set -euo pipefail

REPO_URL="${SKINSCOUT_REPO_URL:-https://github.com/kangk1204/SkinScout_public.git}"
REPO_REF="${SKINSCOUT_REPO_REF:-}"
INSTALL_DIR="${SKINSCOUT_INSTALL_DIR:-${HOME}/SkinScout}"
DRY_RUN=0
NO_LAUNCH=0
WITH_TARGET_MODELS=0
PROFILE="demo"
PROFILE_EXPLICIT=0
ANALOG_BUNDLE=""
ANALOG_BUNDLE_MANIFEST=""
RUNTIME_MODE="conda"

usage() {
    printf '%s\n' \
        "Usage: bash install_skinscout.sh [options]" \
        "" \
        "Options:" \
        "  --install-dir PATH     Installation directory (default: ~/SkinScout)" \
        "  --runtime MODE         Runtime to prepare: conda or compose (default: conda)" \
        "  --profile analog|demo|full  Install profile (default: demo). analog=대체소재 검색만, Stage 0 없음" \
        "  --analog-bundle URL|PATH    Data bundle for --profile analog (required for it)" \
        "  --with-target-models   Deprecated alias for --profile full" \
        "  --no-launch            Do not open Workbench after installation" \
        "  --dry-run              Show actions without changing the computer" \
        "  --help                 Show this help" \
        "" \
        "Environment:" \
        "  SKINSCOUT_REPO_REF=TAG   Pin a release tag or branch instead of the default branch"
}

while (($#)); do
    case "$1" in
        --install-dir) shift; INSTALL_DIR="${1:?--install-dir requires a path}" ;;
        --runtime) shift; RUNTIME_MODE="${1:?--runtime requires compose or conda}" ;;
        --profile) shift; PROFILE="${1:?--profile requires analog, demo or full}"; PROFILE_EXPLICIT=1 ;;
        --analog-bundle) shift; ANALOG_BUNDLE="${1:?--analog-bundle requires a URL or path}" ;;
        --analog-bundle-manifest) shift; ANALOG_BUNDLE_MANIFEST="${1:?--analog-bundle-manifest requires a URL or path}" ;;
        --with-target-models) WITH_TARGET_MODELS=1 ;;
        --no-launch) NO_LAUNCH=1 ;;
        --dry-run) DRY_RUN=1 ;;
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/scripts/run_skinscout.py" ]]; then
    ROOT="${SCRIPT_DIR}"
else
    ROOT="${INSTALL_DIR}"
fi

if [[ ! -r /etc/os-release ]]; then
    printf '[SkinScout][ERROR] Ubuntu 정보를 확인할 수 없습니다.\n' >&2
    exit 2
fi

# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "24.04" || "$(uname -m)" != "x86_64" ]]; then
    if ((DRY_RUN)); then
        log "실제 conda 설치는 Ubuntu 24.04 x86_64에서만 지원합니다."
    elif [[ "${RUNTIME_MODE}" == "conda" ]]; then
        printf '[SkinScout][ERROR] 이 conda 설치기는 Ubuntu 24.04 x86_64 전용입니다.\n' >&2
        exit 2
    fi
fi

missing_packages=()
command -v git >/dev/null 2>&1 || missing_packages+=(git)
command -v curl >/dev/null 2>&1 || missing_packages+=(curl)
command -v bzip2 >/dev/null 2>&1 || missing_packages+=(bzip2)
command -v wget >/dev/null 2>&1 || missing_packages+=(wget)
command -v python3 >/dev/null 2>&1 || missing_packages+=(python3)
command -v tar >/dev/null 2>&1 || missing_packages+=(tar)
command -v sha256sum >/dev/null 2>&1 || missing_packages+=(coreutils)
command -v xdg-open >/dev/null 2>&1 || missing_packages+=(xdg-utils)
command -v clinfo >/dev/null 2>&1 || missing_packages+=(clinfo)
browser_packages=(
    xvfb
    fonts-noto-color-emoji
    fonts-unifont
    libfontconfig1
    libfreetype6
    xfonts-cyrillic
    xfonts-scalable
    fonts-liberation
    fonts-ipafont-gothic
    fonts-wqy-zenhei
    fonts-tlwg-loma-otf
    fonts-freefont-ttf
    libasound2t64
    libatk-bridge2.0-0t64
    libatk1.0-0t64
    libatspi2.0-0t64
    libcairo2
    libcups2t64
    libdbus-1-3
    libdrm2
    libgbm1
    libglib2.0-0t64
    libnspr4
    libnss3
    libpango-1.0-0
    libx11-6
    libxcb1
    libxcomposite1
    libxdamage1
    libxext6
    libxfixes3
    libxkbcommon0
    libxrandr2
    at-spi2-common
    at-spi2-core
    gir1.2-atk-1.0
    gir1.2-atspi-2.0
    libatk-adaptor
    libegl-mesa0
    libgl1-mesa-dri
    libglx-mesa0
    mesa-libgallium
    mesa-vulkan-drivers
)
for package in "${browser_packages[@]}"; do
    if ! dpkg-query -W -f='${Status}' "${package}" 2>/dev/null \
        | grep -qx 'install ok installed'; then
        missing_packages+=("${package}")
    fi
done
if [[ "${RUNTIME_MODE}" == "compose" ]]; then
    command -v gpg >/dev/null 2>&1 || missing_packages+=(gnupg)
    command -v lsb_release >/dev/null 2>&1 || missing_packages+=(lsb-release)
fi

if ((${#missing_packages[@]})); then
    log "Ubuntu 기본 도구를 준비합니다: ${missing_packages[*]}"
    if ((EUID == 0)); then
        run apt-get update
        run apt-get install --yes ca-certificates "${missing_packages[@]}"
    elif command -v sudo >/dev/null 2>&1; then
        run sudo apt-get update
        run sudo apt-get install --yes ca-certificates "${missing_packages[@]}"
    else
        printf '[SkinScout][ERROR] 기본 도구 설치에 관리자 권한이 필요합니다.\n' >&2
        exit 2
    fi
fi

if [[ ! -f "${ROOT}/scripts/run_skinscout.py" ]]; then
    log "SkinScout 소스를 ${ROOT}에 받습니다."
    # SKINSCOUT_REPO_REF pins a release tag or branch so a fleet of hosts can
    # install the same revision. Empty (default) follows the repository default
    # branch, which is what the one-line install does.
    if [[ -n "${REPO_REF}" ]]; then
        run git clone --branch "${REPO_REF}" "${REPO_URL}" "${ROOT}"
    else
        run git clone "${REPO_URL}" "${ROOT}"
    fi
fi

if ((DRY_RUN)) && [[ ! -f "${ROOT}/scripts/bootstrap_runtime.sh" ]]; then
    log "clone 후 scripts/bootstrap_runtime.sh를 실행할 예정입니다."
else
    bootstrap_args=()
    ((DRY_RUN)) && bootstrap_args+=(--dry-run)
    bootstrap_args+=(--runtime "${RUNTIME_MODE}")
    bootstrap_args+=(--profile "${PROFILE}")
    [[ -n "${ANALOG_BUNDLE}" ]] && bootstrap_args+=(--analog-bundle "${ANALOG_BUNDLE}")
    [[ -n "${ANALOG_BUNDLE_MANIFEST}" ]] && bootstrap_args+=(--analog-bundle-manifest "${ANALOG_BUNDLE_MANIFEST}")
    run bash "${ROOT}/scripts/bootstrap_runtime.sh" "${bootstrap_args[@]}"
fi

LOCAL_BIN="${HOME}/.local/bin"
COMMAND_PATH="${LOCAL_BIN}/skinscout"
APPLICATION_DIR="${HOME}/.local/share/applications"
DESKTOP_FILE="${APPLICATION_DIR}/skinscout-workbench.desktop"

if ((DRY_RUN)); then
    log "실행 명령 생성 예정: ${COMMAND_PATH}"
    log "앱 메뉴 바로가기 생성 예정: ${DESKTOP_FILE}"
else
    mkdir -p "${LOCAL_BIN}" "${APPLICATION_DIR}"
    if [[ "${RUNTIME_MODE}" == "compose" ]]; then
        printf '#!/usr/bin/env bash\nexec python3 %q "$@"\n' \
            "${ROOT}/scripts/host_cli.py" > "${COMMAND_PATH}"
    else
        printf '#!/usr/bin/env bash\nexec bash %q "$@"\n' \
            "${ROOT}/scripts/conda_host_cli.sh" > "${COMMAND_PATH}"
    fi
    chmod 0755 "${COMMAND_PATH}"
    {
        printf '%s\n' \
            '[Desktop Entry]' \
            'Type=Application' \
            'Name=SkinScout Workbench' \
            'Comment=피부 관련 화합물 안전성과 단백질 표적을 탐색합니다' \
            "Exec=${COMMAND_PATH} start" \
            'Terminal=false' \
            'Categories=Science;Education;' \
            'StartupNotify=true'
    } > "${DESKTOP_FILE}"
    chmod 0644 "${DESKTOP_FILE}"
fi

log "설치가 끝났습니다. 이후에는 앱 메뉴에서 SkinScout Workbench를 실행할 수 있습니다."
if ((NO_LAUNCH == 0 && DRY_RUN == 0)); then
    exec "${COMMAND_PATH}" start
fi
