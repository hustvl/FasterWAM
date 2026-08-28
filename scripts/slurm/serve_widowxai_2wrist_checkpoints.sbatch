#!/usr/bin/env bash
# The eight newest checkpoints are ordered newest first. Each gets one GPU and
# a low-to-high port in the same order. SERVER_PORT_BASE defaults to 8001, so
# step 1600 -> 8001, ..., step 480 -> 8008.
#SBATCH --job-name=fasterwam-2wrist-serve
#SBATCH --partition=128-partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-7%8
#SBATCH --output=/lustre/data/fasterwam/slurm/%x-%A_%a.out
#SBATCH --error=/lustre/data/fasterwam/slurm/%x-%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/user128004/gitrepo/FasterWAM}"
FASTWAM_REPO="${FASTWAM_REPO:-/home/user128004/gitrepo/FastWAM}"
MODEL_ROOT="${MODEL_ROOT:-/lustre/data/fastwam_nakamura/checkpoints}"
RUN_DIR="${RUN_DIR:-/lustre/data/fasterwam/runs/widowxai_fasterwam_2wrist_vertical_384_1e-4/widowxai_2wrist_vertical_scratch_4443674}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
ACTION_INFER_MODE="${ACTION_INFER_MODE:-one_pass_future_cache}"
SEED="${SEED:-42}"
SERVER_PORT_BASE="${SERVER_PORT_BASE:-8001}"

# Array order is intentionally newest to oldest. Steps 320 and 160 are omitted.
CHECKPOINT_STEPS=(1600 1440 1280 1120 960 800 640 480)
ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:?This launcher must be submitted as a Slurm array}"
if (( ARRAY_INDEX < 0 || ARRAY_INDEX >= ${#CHECKPOINT_STEPS[@]} )); then
  echo "[serve-array] invalid array index: ${ARRAY_INDEX}" >&2
  exit 2
fi

STEP="${CHECKPOINT_STEPS[${ARRAY_INDEX}]}"
CHECKPOINT_TAG="$(printf 'step_%06d' "${STEP}")"
CHECKPOINT="${RUN_DIR}/checkpoints/weights/${CHECKPOINT_TAG}.pt"
# Lower ports are assigned to the more advanced checkpoints.
SERVER_PORT="$((SERVER_PORT_BASE + ARRAY_INDEX))"

for required_path in \
  "${REPO_ROOT}/scripts/serve_widowxai_policy.py" \
  "${FASTWAM_REPO}/scripts/pixi_run.sh" \
  "${RUN_DIR}/config.yaml" \
  "${RUN_DIR}/dataset_stats.json" \
  "${CHECKPOINT}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "[serve-array] missing required path: ${required_path}" >&2
    exit 2
  fi
done

mkdir -p /lustre/data/fasterwam/slurm
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

echo "[serve-array] job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} node=$(hostname)"
echo "[serve-array] checkpoint=${CHECKPOINT} step=${STEP}"
echo "[serve-array] server=0.0.0.0:${SERVER_PORT} gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[serve-array] priority_order=newest_first array_index=${ARRAY_INDEX}"
echo "[serve-array] tunnel=ssh -N -L ${SERVER_PORT}:$(hostname):${SERVER_PORT} <user>@<login-node>"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

exec bash "${FASTWAM_REPO}/scripts/pixi_run.sh" python \
  scripts/serve_widowxai_policy.py \
  --run-dir "${RUN_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --norm-stats "${RUN_DIR}/dataset_stats.json" \
  --model-base-path "${MODEL_ROOT}" \
  --host 0.0.0.0 \
  --port "${SERVER_PORT}" \
  --device cuda:0 \
  --num-inference-steps "${NUM_INFERENCE_STEPS}" \
  --action-infer-mode "${ACTION_INFER_MODE}" \
  --seed "${SEED}"
