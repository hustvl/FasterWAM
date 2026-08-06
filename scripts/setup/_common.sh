#!/usr/bin/env bash
set -euo pipefail

FASTERWAM_SETUP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FASTERWAM_ROOT="$(cd -- "${FASTERWAM_SETUP_DIR}/../.." && pwd)"

fasterwam_find_uv() {
  if command -v uv >/dev/null 2>&1; then
    command -v uv
  else
    echo "uv is required. Install it from https://docs.astral.sh/uv/." >&2
    exit 1
  fi
}

fasterwam_prepare_core_env() {
  local env_dir="$1"
  local uv_bin="$2"
  UV_PROJECT_ENVIRONMENT="${env_dir}" "${uv_bin}" sync \
    --project "${FASTERWAM_ROOT}" \
    --python 3.10
}

fasterwam_prepare_benchmark_env() {
  local project_dir="$1"
  local env_dir="$2"
  local uv_bin="$3"
  if [ ! -f "${project_dir}/uv.lock" ]; then
    echo "Missing benchmark lock file: ${project_dir}/uv.lock" >&2
    exit 1
  fi
  UV_PROJECT_ENVIRONMENT="${env_dir}" "${uv_bin}" sync \
    --project "${project_dir}" \
    --python 3.10
}

fasterwam_clone_pinned() {
  local repo_url="$1"
  local revision="$2"
  local target="$3"
  if [ -d "${target}" ]; then
    if ! git -C "${target}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
      echo "Cannot verify source revision because this is not a Git checkout: ${target}" >&2
      exit 1
    fi
    local actual_revision
    local expected_revision
    actual_revision="$(git -C "${target}" rev-parse HEAD)"
    expected_revision="$(git -C "${target}" rev-parse "${revision}^{commit}")"
    if [ "${actual_revision}" != "${expected_revision}" ]; then
      echo "Source revision mismatch for ${target}" >&2
      echo "  expected: ${revision} (${expected_revision})" >&2
      echo "  actual:   ${actual_revision}" >&2
      exit 1
    fi
    return 0
  fi
  mkdir -p "$(dirname "${target}")"
  git clone "${repo_url}" "${target}"
  git -C "${target}" checkout "${revision}"
}
