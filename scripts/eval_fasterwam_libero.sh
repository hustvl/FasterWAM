#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

: "${CKPT_PATH:?Set CKPT_PATH to a FasterWAM weights checkpoint.}"
: "${DATASET_STATS_PATH:?Set DATASET_STATS_PATH to dataset_stats.json.}"

PYTHON_BIN="${REPO_ROOT}/.venvs/libero/bin/python"
LIBERO_SOURCE_DIR="${REPO_ROOT}/third_party/LIBERO"
if [ ! -x "${PYTHON_BIN}" ]; then
  echo "LIBERO Python is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [ ! -d "${LIBERO_SOURCE_DIR}" ]; then
  echo "LIBERO source is missing: ${LIBERO_SOURCE_DIR}" >&2
  exit 1
fi
NUM_GPUS="${NUM_GPUS:-8}"
MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-2}"
TASK_NAME="${TASK_NAME:-libero_fasterwam_2cam224_1e-4}"

export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${REPO_ROOT}/.runtime/numba/libero}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.runtime/matplotlib/libero}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${REPO_ROOT}/.runtime/libero}"
export PYTHONPATH="${LIBERO_SOURCE_DIR}:${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}"

"${PYTHON_BIN}" scripts/setup/configure_libero.py \
  --source-root "${LIBERO_SOURCE_DIR}" \
  --config-dir "${LIBERO_CONFIG_PATH}"

"${PYTHON_BIN}" experiments/libero/run_libero_manager.py \
  "task=${TASK_NAME}" \
  "ckpt=${CKPT_PATH}" \
  "EVALUATION.dataset_stats_path=${DATASET_STATS_PATH}" \
  "MULTIRUN.num_gpus=${NUM_GPUS}" \
  "MULTIRUN.max_tasks_per_gpu=${MAX_TASKS_PER_GPU}" \
  "$@"
