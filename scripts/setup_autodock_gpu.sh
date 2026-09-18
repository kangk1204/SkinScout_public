#!/usr/bin/env bash
# Build and install AutoDock-GPU into the repo-local ignored tools/ tree.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_PREFIX="${SKINSCOUT_AUTODOCK_BUILD_PREFIX:-/tmp/skinscout-autodock-build}"
SRC_DIR="${SKINSCOUT_AUTODOCK_SRC_DIR:-${ROOT}/tools/AutoDock-GPU}"
INSTALL_DIR="${SKINSCOUT_AUTODOCK_INSTALL_DIR:-${ROOT}/tools/autodock_gpu}"
NUMWI="${SKINSCOUT_AUTODOCK_NUMWI:-128}"
REPO_URL="${SKINSCOUT_AUTODOCK_REPO_URL:-https://github.com/ccsb-scripps/AutoDock-GPU.git}"
REPO_COMMIT="${SKINSCOUT_AUTODOCK_REPO_COMMIT:-6b150b35d0c615bc8dcec40ae09ca855227cc66f}"

if ! command -v micromamba >/dev/null 2>&1; then
    echo "[setup.autodock_gpu][FATAL] micromamba is required on PATH" >&2
    exit 2
fi
if [ "${NUMWI}" != "128" ]; then
    echo "[setup.autodock_gpu][FATAL] SkinScout expects NUMWI=128 for autodock_gpu_128wi" >&2
    exit 2
fi
if [[ ! "${REPO_COMMIT}" =~ ^[0-9a-fA-F]{40}$ ]]; then
    echo "[setup.autodock_gpu][FATAL] SKINSCOUT_AUTODOCK_REPO_COMMIT must be a full 40-character commit" >&2
    exit 2
fi

mkdir -p "${ROOT}/tools"
if [ ! -d "${SRC_DIR}/.git" ]; then
    git clone --no-checkout "${REPO_URL}" "${SRC_DIR}"
    git -C "${SRC_DIR}" fetch --depth 1 origin "${REPO_COMMIT}"
    git -C "${SRC_DIR}" checkout --detach "${REPO_COMMIT}"
fi
RESOLVED_COMMIT="$(git -C "${SRC_DIR}" rev-parse HEAD)"
if [ "${RESOLVED_COMMIT}" != "${REPO_COMMIT}" ]; then
    echo "[setup.autodock_gpu][FATAL] source checkout ${RESOLVED_COMMIT} does not match pinned commit ${REPO_COMMIT}" >&2
    exit 2
fi
if [ -n "$(git -C "${SRC_DIR}" status --porcelain --untracked-files=all)" ]; then
    echo "[setup.autodock_gpu][FATAL] pinned source checkout contains local modifications or untracked files" >&2
    exit 2
fi

if [ -d "${BUILD_PREFIX}" ]; then
    micromamba install -y -p "${BUILD_PREFIX}" -c conda-forge \
        cxx-compiler make git opencl-headers ocl-icd patchelf
else
    micromamba create -y -p "${BUILD_PREFIX}" -c conda-forge \
        cxx-compiler make git opencl-headers ocl-icd patchelf
fi

micromamba run -p "${BUILD_PREFIX}" make \
    DEVICE=OCLGPU \
    NUMWI="${NUMWI}" \
    GPU_INCLUDE_PATH="${BUILD_PREFIX}/include" \
    GPU_LIBRARY_PATH="${BUILD_PREFIX}/lib" \
    -C "${SRC_DIR}"

mkdir -p "${INSTALL_DIR}/bin" "${INSTALL_DIR}/lib" "${INSTALL_DIR}/libexec"
cp "${SRC_DIR}/bin/autodock_gpu_${NUMWI}wi" \
    "${INSTALL_DIR}/libexec/autodock_gpu_128wi.real"
cp "${BUILD_PREFIX}/lib/libstdc++.so.6" \
   "${BUILD_PREFIX}/lib/libgomp.so.1" \
   "${BUILD_PREFIX}/lib/libgcc_s.so.1" \
   "${INSTALL_DIR}/lib/"

micromamba run -p "${BUILD_PREFIX}" patchelf \
    --set-rpath '$ORIGIN/../lib' \
    "${INSTALL_DIR}/libexec/autodock_gpu_128wi.real"

cat > "${INSTALL_DIR}/bin/autodock_gpu_128wi" <<'WRAPPER'
#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
REAL="${ROOT}/libexec/autodock_gpu_128wi.real"

export LD_LIBRARY_PATH="${ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [ -r /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 ]; then
    export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libOpenCL.so.1${LD_PRELOAD:+ ${LD_PRELOAD}}"
fi

exec "${REAL}" "$@"
WRAPPER
chmod +x "${INSTALL_DIR}/bin/autodock_gpu_128wi" \
    "${INSTALL_DIR}/libexec/autodock_gpu_128wi.real"

"${INSTALL_DIR}/bin/autodock_gpu_128wi" --help >/dev/null
python "${ROOT}/scripts/model_readiness.py" --require autodock_gpu >/dev/null
cat > "${INSTALL_DIR}/source_manifest.json" <<EOF
{
  "repository": "${REPO_URL}",
  "commit": "${RESOLVED_COMMIT}",
  "num_work_items": ${NUMWI}
}
EOF
echo "[setup.autodock_gpu] installed ${INSTALL_DIR}/bin/autodock_gpu_128wi"
