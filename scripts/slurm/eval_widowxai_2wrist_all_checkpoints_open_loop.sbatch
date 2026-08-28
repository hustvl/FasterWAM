#!/usr/bin/env bash
#SBATCH --job-name=fasterwam-2wrist-ol
#SBATCH --partition=128-partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --array=0-6%1
#SBATCH --output=/lustre/data/fasterwam/slurm/%x-%A_%a.out
#SBATCH --error=/lustre/data/fasterwam/slurm/%x-%A_%a.err

set -euo pipefail

TRAINING_JOB_ID="4443674"
EVAL_PHASE="${EVAL_PHASE:-initial}"
RESUME_JOB_ID="${RESUME_JOB_ID:-unset}"
TRAIN_REPO="/home/user128004/gitrepo/FasterWAM-widowxai-train"
EVAL_SCRIPT="/home/user128004/gitrepo/FasterWAM/scripts/eval_widowxai_open_loop.py"
FASTWAM_RUNNER="/home/user128004/gitrepo/FastWAM/scripts/pixi_run.sh"
DATA_ROOT="/lustre/data/fastwam_nakamura"
RESULT_ROOT="/lustre/data/fasterwam"
RUN_ROOT="${RESULT_ROOT}/runs/widowxai_fasterwam_2wrist_vertical_384_1e-4/widowxai_2wrist_vertical_scratch_${TRAINING_JOB_ID}"

case "${EVAL_PHASE}" in
  initial)
    CHECKPOINT_STEPS=(160 320 480 640 800 960 1120)
    OUTPUT_PHASE="training_job_${TRAINING_JOB_ID}"
    ;;
  post_resume)
    if [[ "${RESUME_JOB_ID}" == "unset" ]]; then
      echo "[fasterwam_2wrist_open_loop] RESUME_JOB_ID is required for EVAL_PHASE=post_resume" >&2
      exit 2
    fi
    CHECKPOINT_STEPS=(1120 1280 1440 1600)
    OUTPUT_PHASE="training_job_${TRAINING_JOB_ID}_post_resume_job_${RESUME_JOB_ID}"
    ;;
  *)
    echo "[fasterwam_2wrist_open_loop] unsupported EVAL_PHASE=${EVAL_PHASE}" >&2
    exit 2
    ;;
esac

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#CHECKPOINT_STEPS[@]} )); then
  echo "[fasterwam_2wrist_open_loop] array index ${SLURM_ARRAY_TASK_ID} is out of range for ${EVAL_PHASE}" >&2
  exit 2
fi

STEP="${CHECKPOINT_STEPS[${SLURM_ARRAY_TASK_ID}]}"
CHECKPOINT_TAG="$(printf 'step_%06d' "${STEP}")"
CHECKPOINT="${RUN_ROOT}/checkpoints/weights/${CHECKPOINT_TAG}.pt"
NORM_STATS="${NORM_STATS:-${DATA_ROOT}/dataset_stats.json}"
Q01_Q99_STATS="${Q01_Q99_STATS:-/lustre/data/dreamzero/real_widowxai_success_only_fix_prompt/meta/relative_stats_dreamzero.json}"
COMPARISON_ACTION_HORIZON="${COMPARISON_ACTION_HORIZON:-24}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
ACTION_INFER_MODE="${ACTION_INFER_MODE:-one_pass_future_cache}"
SEED="${SEED:-42}"
OUTPUT_ROOT="${RESULT_ROOT}/evaluations/widowxai_2wrist_vertical_open_loop/${OUTPUT_PHASE}/eval_job_${SLURM_ARRAY_JOB_ID}"
OUTPUT_DIR="${OUTPUT_ROOT}/${CHECKPOINT_TAG}"
CONFIG_PATH="${OUTPUT_DIR}/resolved_config.yaml"

for required_path in \
  "${TRAIN_REPO}/scripts/train.py" \
  "${EVAL_SCRIPT}" \
  "${FASTWAM_RUNNER}" \
  "${CHECKPOINT}" \
  "${NORM_STATS}" \
  "${Q01_Q99_STATS}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "[fasterwam_2wrist_open_loop] missing required path: ${required_path}" >&2
    exit 2
  fi
done

cd "${TRAIN_REPO}"
mkdir -p "${OUTPUT_DIR}"

export FASTWAM_DATA_ROOT="${DATA_ROOT}"
export DIFFSYNTH_MODEL_BASE_PATH="${DATA_ROOT}/checkpoints"
export PYTHONPATH="${TRAIN_REPO}/src:${TRAIN_REPO}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[fasterwam_2wrist_open_loop] training_job=${TRAINING_JOB_ID}"
echo "[fasterwam_2wrist_open_loop] eval_phase=${EVAL_PHASE} resume_job=${RESUME_JOB_ID}"
echo "[fasterwam_2wrist_open_loop] array_job=${SLURM_ARRAY_JOB_ID} array_task=${SLURM_ARRAY_TASK_ID}"
echo "[fasterwam_2wrist_open_loop] checkpoint=${CHECKPOINT}"
echo "[fasterwam_2wrist_open_loop] output=${OUTPUT_DIR}"
echo "[fasterwam_2wrist_open_loop] mode=${ACTION_INFER_MODE} steps=${NUM_INFERENCE_STEPS} seed=${SEED}"
echo "[fasterwam_2wrist_open_loop] dataset=/lustre/data/dreamzero/real_widowxai_test coverage=all"
echo "[fasterwam_2wrist_open_loop] video_metrics=disabled"

# Build the exact two-wrist evaluation config. Pretrained model loads are
# skipped because the complete trained MoT is loaded from CHECKPOINT below.
bash "${FASTWAM_RUNNER}" \
  python scripts/train.py \
  task=widowxai_fasterwam_2wrist_vertical_384_1e-4 \
  model.skip_dit_load_from_pretrain=true \
  model.action_dit_pretrained_path=null \
  model.mot_checkpoint_mixed_attn=false \
  wandb.enabled=false \
  --cfg job --resolve > "${CONFIG_PATH}"

bash "${FASTWAM_RUNNER}" \
  torchrun --standalone --nproc_per_node=8 \
  "${EVAL_SCRIPT}" \
  --config "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --norm-stats "${NORM_STATS}" \
  --q01-q99-stats "${Q01_Q99_STATS}" \
  --comparison-action-horizon "${COMPARISON_ACTION_HORIZON}" \
  --num-inference-steps "${NUM_INFERENCE_STEPS}" \
  --action-infer-mode "${ACTION_INFER_MODE}" \
  --seed "${SEED}"

test -s "${OUTPUT_DIR}/summary.json"
echo "[fasterwam_2wrist_open_loop] completed summary=${OUTPUT_DIR}/summary.json"
