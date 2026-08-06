#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"
UV_BIN="$(fasterwam_find_uv)"
CORE_ENV="${FASTERWAM_ROOT}/.venv"
fasterwam_prepare_core_env "${CORE_ENV}" "${UV_BIN}"
echo "CORE_PYTHON=${CORE_ENV}/bin/python"
