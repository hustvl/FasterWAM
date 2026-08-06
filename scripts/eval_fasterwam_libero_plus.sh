#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

: "${CKPT_PATH:?Set CKPT_PATH to the FasterWAM LIBERO checkpoint.}"
: "${DATASET_STATS_PATH:?Set DATASET_STATS_PATH to dataset_stats.json.}"

PYTHON_BIN="${REPO_ROOT}/.venvs/libero-plus/bin/python"
LIBERO_PLUS_SOURCE_DIR="${REPO_ROOT}/third_party/LIBERO-plus"
if [ ! -x "${PYTHON_BIN}" ]; then
  echo "LIBERO-Plus Python is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [ ! -d "${LIBERO_PLUS_SOURCE_DIR}" ]; then
  echo "LIBERO-Plus source is missing: ${LIBERO_PLUS_SOURCE_DIR}" >&2
  exit 1
fi
NUM_GPUS="${NUM_GPUS:-8}"
TASK_NAME="${TASK_NAME:-libero_fasterwam_2cam224_1e-4}"

export LIBERO_CONFIG_PATH="${LIBERO_PLUS_CONFIG_PATH:-${REPO_ROOT}/.runtime/libero-plus}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${REPO_ROOT}/.runtime/numba/libero-plus}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.runtime/matplotlib/libero-plus}"
export PYTHONPATH="${LIBERO_PLUS_SOURCE_DIR}:${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}"

"${PYTHON_BIN}" scripts/setup/configure_libero.py \
  --source-root "${LIBERO_PLUS_SOURCE_DIR}" \
  --config-dir "${LIBERO_CONFIG_PATH}"

"${PYTHON_BIN}" experiments/libero/run_libero_manager.py \
  --config-name sim_libero_plus \
  "task=${TASK_NAME}" \
  "ckpt=${CKPT_PATH}" \
  "EVALUATION.dataset_stats_path=${DATASET_STATS_PATH}" \
  "MULTIRUN.num_gpus=${NUM_GPUS}" \
  "$@"
