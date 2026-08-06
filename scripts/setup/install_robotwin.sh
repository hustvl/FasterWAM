#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"
UV_BIN="$(fasterwam_find_uv)"
ROBOTWIN_SOURCE_DIR="${FASTERWAM_ROOT}/third_party/RoboTwin"
ROBOTWIN_ENV="${FASTERWAM_ROOT}/.venvs/robotwin"
if [ ! -d "${ROBOTWIN_SOURCE_DIR}" ]; then
  echo "RoboTwin source is missing: ${ROBOTWIN_SOURCE_DIR}" >&2
  exit 1
fi

fasterwam_prepare_benchmark_env \
  "${FASTERWAM_ROOT}/environments/robotwin" "${ROBOTWIN_ENV}" "${UV_BIN}"
"${ROBOTWIN_ENV}/bin/python" "${FASTERWAM_ROOT}/scripts/setup/patch_robotwin_env.py"

CUROBO_ROOT="${ROBOTWIN_SOURCE_DIR}/envs/curobo"
fasterwam_clone_pinned \
  "https://github.com/NVlabs/curobo.git" \
  "v0.7.8" \
  "${CUROBO_ROOT}"
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ] && \
   ! "${ROBOTWIN_ENV}/bin/python" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
  echo "No build GPU is visible. Set TORCH_CUDA_ARCH_LIST for the target GPU architecture." >&2
  exit 1
fi
MAX_JOBS="${MAX_JOBS:-8}" CMAKE_BUILD_PARALLEL_LEVEL="${MAX_JOBS:-8}" \
  "${UV_BIN}" pip install --python "${ROBOTWIN_ENV}/bin/python" \
  --no-deps --no-build-isolation -e "${CUROBO_ROOT}"

if [ ! -d "${ROBOTWIN_SOURCE_DIR}/assets/background_texture" ] || \
   [ ! -d "${ROBOTWIN_SOURCE_DIR}/assets/embodiments" ] || \
   [ ! -d "${ROBOTWIN_SOURCE_DIR}/assets/objects" ]; then
  (cd "${ROBOTWIN_SOURCE_DIR}" && \
    PATH="${ROBOTWIN_ENV}/bin:${PATH}" bash script/_download_assets.sh)
fi

ln -sfn "${FASTERWAM_ROOT}/experiments/robotwin/fasterwam_policy" \
  "${ROBOTWIN_SOURCE_DIR}/policy/fasterwam_policy"
echo "ROBOTWIN_PYTHON=${ROBOTWIN_ENV}/bin/python"
echo "ROBOTWIN_SOURCE_DIR=${ROBOTWIN_SOURCE_DIR}"
