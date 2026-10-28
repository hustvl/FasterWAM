#!/usr/bin/env bash
# Legacy WidowXAI three-camera servers, newest checkpoint first:
#   step 1280 -> port 8001, ..., step 480 -> port 8006.
# To start after the two-wrist server array exits or is cancelled:
#   sbatch --dependency=afterany:<two-wrist-array-job-id> \
#     scripts/slurm/serve_widowxai_3cam_checkpoints.sbatch
#SBATCH --job-name=fasterwam-3cam-serve
#SBATCH --partition=128-partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-5%6
#SBATCH --output=/lustre/data/fasterwam/slurm/%x-%A_%a.out
#SBATCH --error=/lustre/data/fasterwam/slurm/%x-%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/user128004/gitrepo/FasterWAM}"
FASTWAM_REPO="${FASTWAM_REPO:-/home/user128004/gitrepo/FastWAM}"
MODEL_ROOT="${MODEL_ROOT:-/lustre/data/fastwam_nakamura/checkpoints}"
WEIGHT_DIR="${WEIGHT_DIR:-/lustre/data/fasterwam/weight_3cam}"
LEGACY_EVAL_DIR="${LEGACY_EVAL_DIR:-/lustre/data/fasterwam/evaluations/widowxai_open_loop/step_001280/2026-08-21_18-50-06_4400752}"
CONFIG_PATH="${CONFIG_PATH:-${LEGACY_EVAL_DIR}/resolved_config.yaml}"
NORM_STATS="${NORM_STATS:-${LEGACY_EVAL_DIR}/dataset_stats.json}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
ACTION_INFER_MODE="${ACTION_INFER_MODE:-one_pass_future_cache}"
SEED="${SEED:-42}"

# Lower ports go to later steps. Steps 320 and 160 are intentionally omitted.
CHECKPOINT_STEPS=(1280 1120 960 800 640 480)
ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:?This launcher must be submitted as a Slurm array}"
if (( ARRAY_INDEX < 0 || ARRAY_INDEX >= ${#CHECKPOINT_STEPS[@]} )); then
  echo "[serve-3cam-array] invalid array index: ${ARRAY_INDEX}" >&2
  exit 2
fi

STEP="${CHECKPOINT_STEPS[${ARRAY_INDEX}]}"
CHECKPOINT_TAG="$(printf 'step_%06d' "${STEP}")"
CHECKPOINT="${WEIGHT_DIR}/${CHECKPOINT_TAG}.pt"
SERVER_PORT="$((8001 + ARRAY_INDEX))"

for required_path in \
  "${REPO_ROOT}/scripts/serve_widowxai_policy.py" \
  "${FASTWAM_REPO}/scripts/pixi_run.sh" \
  "${CONFIG_PATH}" \
  "${NORM_STATS}" \
  "${CHECKPOINT}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "[serve-3cam-array] missing required path: ${required_path}" >&2
    exit 2
  fi
done

# A config without drop_top_view is deliberately interpreted as the legacy
# three-camera model. Refuse an accidental two-wrist config before GPU loading.
if rg -q '^[[:space:]]*drop_top_view:[[:space:]]*true' "${CONFIG_PATH}"; then
  echo "[serve-3cam-array] config unexpectedly enables drop_top_view: ${CONFIG_PATH}" >&2
  exit 2
fi

mkdir -p /lustre/data/fasterwam/slurm
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

echo "[serve-3cam-array] job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} node=$(hostname)"
echo "[serve-3cam-array] checkpoint=${CHECKPOINT} step=${STEP}"
echo "[serve-3cam-array] config=${CONFIG_PATH}"
echo "[serve-3cam-array] mode=3camera_robotwin drop_top_view=false"
echo "[serve-3cam-array] server=0.0.0.0:${SERVER_PORT} gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[serve-3cam-array] tunnel=ssh -N -L ${SERVER_PORT}:$(hostname):${SERVER_PORT} <user>@<login-node>"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

exec bash "${FASTWAM_REPO}/scripts/pixi_run.sh" python \
  scripts/serve_widowxai_policy.py \
  --run-dir "${WEIGHT_DIR}" \
  --config "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT}" \
  --norm-stats "${NORM_STATS}" \
  --model-base-path "${MODEL_ROOT}" \
  --host 0.0.0.0 \
  --port "${SERVER_PORT}" \
  --device cuda:0 \
  --num-inference-steps "${NUM_INFERENCE_STEPS}" \
  --action-infer-mode "${ACTION_INFER_MODE}" \
  --seed "${SEED}"
