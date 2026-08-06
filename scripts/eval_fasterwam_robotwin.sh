#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

: "${CKPT_PATH:?Set CKPT_PATH to a FasterWAM weights checkpoint.}"
: "${DATASET_STATS_PATH:?Set DATASET_STATS_PATH to dataset_stats.json.}"

PYTHON_BIN="${REPO_ROOT}/.venvs/robotwin/bin/python"
ROBOTWIN_SOURCE_DIR="${REPO_ROOT}/third_party/RoboTwin"
if [ ! -x "${PYTHON_BIN}" ]; then
  echo "RoboTwin Python is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [ ! -d "${ROBOTWIN_SOURCE_DIR}" ]; then
  echo "RoboTwin source is missing: ${ROBOTWIN_SOURCE_DIR}" >&2
  exit 1
fi
NUM_GPUS="${NUM_GPUS:-8}"
MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-2}"
REPLAN_STEPS="${REPLAN_STEPS:-28}"
TASK_NAME="${TASK_NAME:-robotwin_fasterwam_3cam_384_1e-4}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.runtime/matplotlib/robotwin}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${REPO_ROOT}/.runtime/numba/robotwin}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [ -z "${VK_ICD_FILENAMES:-}" ] && [ -f /etc/vulkan/icd.d/nvidia_icd.json ]; then
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

if [ ! -e third_party/RoboTwin/policy/fasterwam_policy ]; then
  echo "Missing RoboTwin policy adapter: third_party/RoboTwin/policy/fasterwam_policy" >&2
  exit 1
fi

"${PYTHON_BIN}" experiments/robotwin/run_robotwin_manager.py \
  "task=${TASK_NAME}" \
  "ckpt=${CKPT_PATH}" \
  "EVALUATION.dataset_stats_path=${DATASET_STATS_PATH}" \
  "EVALUATION.replan_steps=${REPLAN_STEPS}" \
  "EVALUATION.robotwin_root=${ROBOTWIN_SOURCE_DIR}" \
  "MULTIRUN.num_gpus=${NUM_GPUS}" \
  "MULTIRUN.max_tasks_per_gpu=${MAX_TASKS_PER_GPU}" \
  "$@"
