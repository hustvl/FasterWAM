#!/usr/bin/env bash
#SBATCH --job-name=fasterwam-widow-server
#SBATCH --partition=128-partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=/lustre/data/fasterwam/slurm/%x-%j.out
#SBATCH --error=/lustre/data/fasterwam/slurm/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/user128004/gitrepo/FasterWAM}"
FASTWAM_REPO="${FASTWAM_REPO:-/home/user128004/gitrepo/FastWAM}"
MODEL_ROOT="${MODEL_ROOT:-/lustre/data/fastwam_nakamura/checkpoints}"
RUN_DIR="${RUN_DIR:-/lustre/data/fasterwam/runs/widowxai_fasterwam_2wrist_vertical_384_1e-4/widowxai_2wrist_vertical_scratch_4443674}"
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
SERVER_PORT="${SERVER_PORT:-8000}"
DEVICE="${DEVICE:-cuda:0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
ACTION_INFER_MODE="${ACTION_INFER_MODE:-one_pass_future_cache}"
SEED="${SEED:-42}"

cd "${REPO_ROOT}"
mkdir -p /lustre/data/fasterwam/slurm
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

ARGS=(
  scripts/serve_widowxai_policy.py
  --run-dir "${RUN_DIR}"
  --model-base-path "${MODEL_ROOT}"
  --host "${SERVER_HOST}"
  --port "${SERVER_PORT}"
  --device "${DEVICE}"
  --num-inference-steps "${NUM_INFERENCE_STEPS}"
  --action-infer-mode "${ACTION_INFER_MODE}"
  --seed "${SEED}"
)

if [[ -n "${CHECKPOINT:-}" ]]; then
  ARGS+=(--checkpoint "${CHECKPOINT}")
fi
if [[ -n "${NORM_STATS:-}" ]]; then
  ARGS+=(--norm-stats "${NORM_STATS}")
fi
if [[ -n "${TASK:-}" ]]; then
  ARGS+=(--task "${TASK}")
fi

echo "[slurm] job_id=${SLURM_JOB_ID:-local} node=$(hostname)"
echo "[slurm] run_dir=${RUN_DIR} server=${SERVER_HOST}:${SERVER_PORT}"
echo "[slurm] visual_input=resolved_from_config action_infer_mode=${ACTION_INFER_MODE}"

exec bash "${FASTWAM_REPO}/scripts/pixi_run.sh" python "${ARGS[@]}"
