#!/usr/bin/env bash
#SBATCH --job-name=fasterwam-widow-ol
#SBATCH --partition=128-partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/data/fasterwam/slurm/%x-%j.out
#SBATCH --error=/lustre/data/fasterwam/slurm/%x-%j.err

set -euo pipefail

REPO_ROOT="/home/user128004/gitrepo/FasterWAM"
FASTWAM_RUNNER="/home/user128004/gitrepo/FastWAM/scripts/pixi_run.sh"
DATA_ROOT="/lustre/data/fastwam_nakamura"
RESULT_ROOT="/lustre/data/fasterwam"
CHECKPOINT="${CHECKPOINT:-${RESULT_ROOT}/checkpoints/weights/step_001280.pt}"
CHECKPOINT_TAG="$(basename "${CHECKPOINT}" .pt)"
NORM_STATS="${NORM_STATS:-${DATA_ROOT}/dataset_stats.json}"
Q01_Q99_STATS="${Q01_Q99_STATS:-/lustre/data/dreamzero/real_widowxai_success_only_fix_prompt/meta/relative_stats_dreamzero.json}"
COMPARISON_ACTION_HORIZON="${COMPARISON_ACTION_HORIZON:-24}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
ACTION_INFER_MODE="${ACTION_INFER_MODE:-one_pass_future_cache}"
SEED="${SEED:-42}"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)_${SLURM_JOB_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULT_ROOT}/evaluations/widowxai_open_loop/${CHECKPOINT_TAG}/${RUN_ID}}"
CONFIG_PATH="${OUTPUT_DIR}/resolved_config.yaml"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

export FASTWAM_DATA_ROOT="${DATA_ROOT}"
export DIFFSYNTH_MODEL_BASE_PATH="${DATA_ROOT}/checkpoints"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[fasterwam_open_loop] checkpoint=${CHECKPOINT}"
echo "[fasterwam_open_loop] output=${OUTPUT_DIR}"
echo "[fasterwam_open_loop] mode=${ACTION_INFER_MODE} steps=${NUM_INFERENCE_STEPS} seed=${SEED}"
echo "[fasterwam_open_loop] dataset=real_widowxai_test coverage=all"

bash "${FASTWAM_RUNNER}" \
  python scripts/train.py \
  task=widowxai_fasterwam_3cam_384_1e-4 \
  --cfg job --resolve > "${CONFIG_PATH}"

bash "${FASTWAM_RUNNER}" \
  torchrun --standalone --nproc_per_node=8 \
  scripts/eval_widowxai_open_loop.py \
  --config "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --norm-stats "${NORM_STATS}" \
  --q01-q99-stats "${Q01_Q99_STATS}" \
  --comparison-action-horizon "${COMPARISON_ACTION_HORIZON}" \
  --num-inference-steps "${NUM_INFERENCE_STEPS}" \
  --action-infer-mode "${ACTION_INFER_MODE}" \
  --seed "${SEED}"
