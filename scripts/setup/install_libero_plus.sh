#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"
UV_BIN="$(fasterwam_find_uv)"
LIBERO_PLUS_SOURCE_DIR="${FASTERWAM_ROOT}/third_party/LIBERO-plus"
LIBERO_PLUS_ENV="${FASTERWAM_ROOT}/.venvs/libero-plus"
fasterwam_clone_pinned \
  "https://github.com/sylvestf/LIBERO-plus.git" \
  "4976dc3" \
  "${LIBERO_PLUS_SOURCE_DIR}"
fasterwam_prepare_benchmark_env \
  "${FASTERWAM_ROOT}/environments/libero-plus" "${LIBERO_PLUS_ENV}" "${UV_BIN}"
"${UV_BIN}" pip install --python "${LIBERO_PLUS_ENV}/bin/python" \
  --no-deps -e "${LIBERO_PLUS_SOURCE_DIR}"

ASSETS_DIR="${LIBERO_PLUS_SOURCE_DIR}/libero/libero/assets"
NESTED_ASSETS_DIR="${LIBERO_PLUS_SOURCE_DIR}/libero/libero/inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/release/dataset/LIBERO-plus-0/assets"
if [ ! -d "${ASSETS_DIR}" ]; then
  ARCHIVE="${LIBERO_PLUS_SOURCE_DIR}/libero/libero/assets.zip"
  if [ ! -f "${ARCHIVE}" ]; then
    "${LIBERO_PLUS_ENV}/bin/huggingface-cli" download Sylvest/LIBERO-plus \
      assets.zip --repo-type dataset --local-dir "${LIBERO_PLUS_SOURCE_DIR}/libero/libero"
  fi
  unzip -q -o "${ARCHIVE}" -d "${LIBERO_PLUS_SOURCE_DIR}/libero/libero"
  if [ -d "${NESTED_ASSETS_DIR}" ]; then
    mv "${NESTED_ASSETS_DIR}" "${ASSETS_DIR}"
  fi
fi

"${LIBERO_PLUS_ENV}/bin/python" "${FASTERWAM_ROOT}/scripts/setup/configure_libero.py" \
  --source-root "${LIBERO_PLUS_SOURCE_DIR}" \
  --config-dir "${FASTERWAM_ROOT}/.runtime/libero-plus"
echo "LIBERO_PLUS_PYTHON=${LIBERO_PLUS_ENV}/bin/python"
echo "LIBERO_PLUS_SOURCE_DIR=${LIBERO_PLUS_SOURCE_DIR}"
