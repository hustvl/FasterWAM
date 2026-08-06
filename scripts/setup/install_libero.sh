#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"
UV_BIN="$(fasterwam_find_uv)"
LIBERO_SOURCE_DIR="${FASTERWAM_ROOT}/third_party/LIBERO"
LIBERO_ENV="${FASTERWAM_ROOT}/.venvs/libero"
fasterwam_clone_pinned \
  "https://github.com/Lifelong-Robot-Learning/LIBERO.git" \
  "8f1084e" \
  "${LIBERO_SOURCE_DIR}"
fasterwam_prepare_benchmark_env \
  "${FASTERWAM_ROOT}/environments/libero" "${LIBERO_ENV}" "${UV_BIN}"
"${UV_BIN}" pip install --python "${LIBERO_ENV}/bin/python" \
  --no-deps -e "${LIBERO_SOURCE_DIR}"
"${LIBERO_ENV}/bin/python" "${FASTERWAM_ROOT}/scripts/setup/configure_libero.py" \
  --source-root "${LIBERO_SOURCE_DIR}" \
  --config-dir "${FASTERWAM_ROOT}/.runtime/libero"
echo "LIBERO_PYTHON=${LIBERO_ENV}/bin/python"
echo "LIBERO_SOURCE_DIR=${LIBERO_SOURCE_DIR}"
