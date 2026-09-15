#!/usr/bin/env bash
# setup_p2rank.sh
# P2Rank is distributed as a standalone Java tarball (NOT on conda).
# Downloads the 2.5 release into tools/p2rank-2.5/ and symlinks `prank` onto
# the active conda env's bin so the Stage 0 rules find it on PATH.
# Requires: a JRE (openjdk>=17 is in envs/base.yml).
# Usage: bash scripts/setup_p2rank.sh [VERSION]

set -euo pipefail

VERSION="${1:-2.5}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="${COSMAX_TOOLS:-${ROOT}/tools}"
URL="https://github.com/rdk/p2rank/releases/download/${VERSION}/p2rank_${VERSION}.tar.gz"
DEST="${TOOLS_DIR}/p2rank_${VERSION}"
ARCHIVE="${TOOLS_DIR}/p2rank_${VERSION}.tar.gz"
case "${VERSION}" in
    2.5)
        EXPECTED_SHA256="${SKINSCOUT_P2RANK_SHA256:-9c21755967450300f2eb052d059ef128d38021626e52e135561c7c4687891751}"
        ;;
    *)
        EXPECTED_SHA256="${SKINSCOUT_P2RANK_SHA256:-}"
        ;;
esac

if [[ ! "${EXPECTED_SHA256}" =~ ^[0-9a-fA-F]{64}$ ]]; then
    echo "[p2rank][FATAL] an explicit 64-character SKINSCOUT_P2RANK_SHA256 is required for version ${VERSION}" >&2
    exit 2
fi

mkdir -p "${TOOLS_DIR}"

if [ ! -f "${ARCHIVE}" ] \
    || ! printf '%s  %s\n' "${EXPECTED_SHA256}" "${ARCHIVE}" | sha256sum --check --status; then
    echo "[p2rank] downloading ${URL}"
    TMP_ARCHIVE="${ARCHIVE}.tmp"
    rm -f "${TMP_ARCHIVE}"
    wget -nv -O "${TMP_ARCHIVE}" "${URL}"
    printf '%s  %s\n' "${EXPECTED_SHA256}" "${TMP_ARCHIVE}" | sha256sum --check --status
    mv "${TMP_ARCHIVE}" "${ARCHIVE}"
fi

if [ ! -x "${DEST}/prank" ]; then
    tar -xzf "${ARCHIVE}" -C "${TOOLS_DIR}"
fi

if [ ! -x "${DEST}/prank" ]; then
    echo "[p2rank][FATAL] prank launcher not found at ${DEST}/prank after extract." >&2
    exit 2
fi

# Expose `prank` on PATH via a WRAPPER (not a symlink): P2Rank's launcher
# resolves its JAR classpath relative to its own location, so a bare symlink
# breaks with "Could not find or load main class". The wrapper execs the real
# launcher in place.
TARGET_BIN="${CONDA_PREFIX:-$HOME/.local}/bin"
mkdir -p "${TARGET_BIN}"
REAL_PRANK="$(readlink -f "${DEST}/prank")"
cat > "${TARGET_BIN}/prank" <<EOF
#!/usr/bin/env bash
exec "${REAL_PRANK}" "\$@"
EOF
chmod +x "${TARGET_BIN}/prank"

ACTUAL_SHA256="$(sha256sum "${ARCHIVE}" | awk '{print $1}')"
if [ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ]; then
    echo "[p2rank][FATAL] archive checksum no longer matches the pinned SHA-256" >&2
    exit 2
fi
cat > "${DEST}/source_manifest.json" <<EOF
{
  "release_url": "${URL}",
  "version": "${VERSION}",
  "sha256": "${ACTUAL_SHA256}"
}
EOF

echo "[p2rank] installed → ${DEST}"
echo "[p2rank] wrapper    → ${TARGET_BIN}/prank"
"${TARGET_BIN}/prank" 2>&1 | head -3 || true
