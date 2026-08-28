#!/usr/bin/env bash
#SBATCH --job-name=fasterwam-2wrist-resume
#SBATCH --partition=128-partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=1-00:00:00
#SBATCH --output=/lustre/data/fasterwam/slurm/%x-%j.out
#SBATCH --error=/lustre/data/fasterwam/slurm/%x-%j.err

set -euo pipefail

JOB_START_EPOCH="$(date +%s)"
on_exit() {
  local status=$?
  local elapsed=$(( $(date +%s) - JOB_START_EPOCH ))
  echo "[job] status=${status} elapsed_seconds=${elapsed} end=$(date -Is)"
  trap - EXIT
  exit "${status}"
}
trap on_exit EXIT

TRAINING_JOB_ID="4443674"
EVALUATION_JOB_ID="4446867"
TRAIN_REPO="/home/user128004/gitrepo/FasterWAM-widowxai-train"
FASTWAM_REPO="/home/user128004/gitrepo/FastWAM"
FASTWAM_DATA_ROOT="/lustre/data/fastwam_nakamura"
RESULT_ROOT="/lustre/data/fasterwam"
MODEL_ROOT="${FASTWAM_DATA_ROOT}/checkpoints"
TRAIN_DATASET="/lustre/data/dreamzero/real_widowxai_success_only"
EVAL_DATASET="/lustre/data/dreamzero/real_widowxai_test"
TEXT_CACHE="${FASTWAM_DATA_ROOT}/text_embeds_cache/widowxai"
STATS_PATH="${FASTWAM_DATA_ROOT}/dataset_stats.json"
ACTION_DIT_PATH="${RESULT_ROOT}/pretrained/SparseActionDiT_cond_0_4_8_12_16_20_24_28_Wan22_alphascale_1024hdim_widowxai.pt"
TASK_CONFIG="widowxai_fasterwam_2wrist_vertical_384_1e-4"

export RUN_ID="widowxai_2wrist_vertical_scratch_${TRAINING_JOB_ID}"
export FASTERWAM_OUTPUT_ROOT="${RESULT_ROOT}/runs"
export FASTWAM_OUTPUT_ROOT="${FASTERWAM_OUTPUT_ROOT}"
export PIXI_HOME="${FASTWAM_DATA_ROOT}/pixi"
export HF_HOME="${FASTWAM_DATA_ROOT}/cache/huggingface"
export TORCH_EXTENSIONS_DIR="${FASTWAM_DATA_ROOT}/cache/torch_extensions"
export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_ROOT}"
export TMPDIR="${RESULT_ROOT}/tmp/${SLURM_JOB_ID}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export WANDB_MODE="online"
unset WANDB_RUN_ID WANDB_RESUME
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"

RUN_DIR="${FASTERWAM_OUTPUT_ROOT}/${TASK_CONFIG}/${RUN_ID}"
RESUME_STATE="${RUN_DIR}/checkpoints/state/step_000960"
INCOMPLETE_STATE="${RUN_DIR}/checkpoints/state/step_001120"
RESOLVED_CONFIG="${RUN_DIR}/resolved_config_resume_${SLURM_JOB_ID}.yaml"

mkdir -p \
  "${RESULT_ROOT}/slurm" \
  "${RESULT_ROOT}/tmp/${SLURM_JOB_ID}"

cd "${TRAIN_REPO}"
ulimit -n 131072

run_in_env() {
  bash "${FASTWAM_REPO}/scripts/pixi_run.sh" \
    env "PYTHONPATH=${TRAIN_REPO}/src" "$@"
}

echo "[resume] start=$(date -Is) host=$(hostname) job=${SLURM_JOB_ID} dependency=${SLURM_JOB_DEPENDENCY:-unset}"
echo "[resume] completed_eval_job=${EVALUATION_JOB_ID} original_training_job=${TRAINING_JOB_ID}"
echo "[resume] code_commit=$(git rev-parse HEAD)"
echo "[resume] run_dir=${RUN_DIR}"
echo "[resume] state=${RESUME_STATE} target_step=1600 global_batch=1024"
echo "[resume] visual_input=drop_top_view left_wrist_top right_wrist_bottom final_size=384x320"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

test "$(git rev-parse HEAD)" = "109e45a659cb12c4757753237c773d90c751b596"
test -x "${PIXI_HOME}/bin/pixi"
test -d "${TRAIN_DATASET}"
test -d "${EVAL_DATASET}"
test -d "${TEXT_CACHE}"
test -s "${STATS_PATH}"
test -s "${ACTION_DIT_PATH}"
test -s "${MODEL_ROOT}/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"

# The failed save at step 1120 has no rank-0 optimizer shard, latest marker,
# scheduler, or trainer state. Validate this exact failure signature before
# removing only that unusable directory so the resumed save can reuse its path.
run_in_env python -u - "${RESUME_STATE}" "${INCOMPLETE_STATE}" <<'PY'
import json
import sys
from pathlib import Path

resume = Path(sys.argv[1]).resolve()
incomplete = Path(sys.argv[2]).resolve()
expected_root = resume.parent.resolve()
assert incomplete.parent == expected_root, (incomplete, expected_root)
assert resume.name == "step_000960", resume
assert incomplete.name == "step_001120", incomplete

required = [
    resume / "latest",
    resume / "scheduler.bin",
    resume / "trainer_state.json",
    resume / "pytorch_model" / "mp_rank_00_model_states.pt",
]
required += [
    resume / "pytorch_model" / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
    for rank in range(8)
]
for path in required:
    assert path.is_file() and path.stat().st_size > 0, path
with (resume / "trainer_state.json").open("r", encoding="utf-8") as f:
    trainer_state = json.load(f)
assert trainer_state == {"global_step": 960, "epoch": 16, "batch_in_epoch": 128}, trainer_state

if incomplete.exists():
    assert incomplete.is_dir(), incomplete
    forbidden = [
        incomplete / "latest",
        incomplete / "scheduler.bin",
        incomplete / "trainer_state.json",
        incomplete / "pytorch_model" / "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt",
    ]
    assert not any(path.exists() for path in forbidden), forbidden
    partial_shards = sorted((incomplete / "pytorch_model").glob("bf16_zero_pp_rank_*_mp_rank_00_optim_states.pt"))
    assert [path.name for path in partial_shards] == [
        f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt" for rank in range(1, 8)
    ], partial_shards
print(f"resume preflight ok: complete={resume} incomplete={incomplete if incomplete.exists() else 'already absent'}")
PY

if [[ -d "${INCOMPLETE_STATE}" ]]; then
  echo "[resume] deleting unusable partial state: ${INCOMPLETE_STATE} ($(du -sh "${INCOMPLETE_STATE}" | cut -f1))"
  rm -rf -- "${INCOMPLETE_STATE}"
fi
test ! -e "${INCOMPLETE_STATE}"

MIN_AVAILABLE_BYTES="$((150 * 1024 * 1024 * 1024))"
AVAILABLE_BYTES=0
# Lustre may update free-space accounting a few seconds after a large removal.
for attempt in $(seq 1 24); do
  AVAILABLE_BYTES="$(df --output=avail -B1 "${RESULT_ROOT}" | tail -1 | tr -d ' ')"
  echo "[resume] space_check=${attempt}/24 available_bytes=${AVAILABLE_BYTES} required_bytes=${MIN_AVAILABLE_BYTES}"
  if (( AVAILABLE_BYTES >= MIN_AVAILABLE_BYTES )); then
    break
  fi
  sleep 5
done
if (( AVAILABLE_BYTES < MIN_AVAILABLE_BYTES )); then
  echo "[resume] insufficient free space after partial-state cleanup" >&2
  exit 3
fi

echo "[resume] prebuilding DeepSpeed CPUAdam"
run_in_env python -u scripts/prebuild_ds_ops.py

TRAIN_OVERRIDES=(
  "task=${TASK_CONFIG}"
  "output_dir=${RUN_DIR}"
  "data.train.dataset_dirs=[${TRAIN_DATASET}]"
  "data.val.dataset_dirs=[${EVAL_DATASET}]"
  "data.train.text_embedding_cache_dir=${TEXT_CACHE}"
  "data.val.text_embedding_cache_dir=${TEXT_CACHE}"
  "data.train.pretrained_norm_stats=${STATS_PATH}"
  "model.action_dit_pretrained_path=${ACTION_DIT_PATH}"
  "model.skip_dit_load_from_pretrain=false"
  "model.redirect_common_files=false"
  "model.mot_checkpoint_mixed_attn=true"
  "model.loss.lambda_video=1.0"
  "model.loss.lambda_action=1.0"
  "batch_size=32"
  "gradient_accumulation_steps=4"
  "learning_rate=1e-4"
  "lr_scheduler_type=cosine"
  "weight_decay=1e-2"
  "max_grad_norm=1.0"
  "num_epochs=28"
  "max_steps=1600"
  "save_every=160"
  "eval_every=0"
  "keep_last_n_states=1"
  "resume=${RESUME_STATE}"
  "seed=42"
  "wandb.enabled=true"
  "wandb.mode=online"
  "wandb.name=${RUN_ID}_scheduler_fix_${SLURM_JOB_ID}"
  "wandb.group=widowxai_2wrist_vertical"
)

run_in_env python -u scripts/train.py \
  "${TRAIN_OVERRIDES[@]}" \
  --cfg job --resolve > "${RESOLVED_CONFIG}"

run_in_env python -u - "${RESOLVED_CONFIG}" "${RESUME_STATE}" <<'PY'
import sys
from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
resume_state = sys.argv[2]
assert cfg.batch_size == 32
assert cfg.gradient_accumulation_steps == 4
assert cfg.batch_size * 8 * cfg.gradient_accumulation_steps == 1024
assert cfg.max_steps == 1600
assert cfg.save_every == 160
assert cfg.eval_every == 0
assert cfg.keep_last_n_states == 1
assert cfg.resume == resume_state
assert cfg.data.train.drop_top_view is True
assert cfg.data.val.drop_top_view is True
assert cfg.data.train.concat_multi_camera == "vertical"
assert [item.key for item in cfg.data.train.shape_meta.images] == [
    "cam_left_wrist", "cam_right_wrist"
]
assert cfg.data.train.processor.num_output_cameras == 2
assert list(cfg.data.train.video_size) == [384, 320]
assert cfg.model.mot_checkpoint_mixed_attn is True
assert cfg.model.loss.lambda_video == 1.0
assert cfg.model.loss.lambda_action == 1.0
print("resolved resume config ok: step=960->1600, global_batch=1024, save_every=160, keep_last_state=1")
PY

sha256sum "${ACTION_DIT_PATH}" "${STATS_PATH}"
echo "[resume] launching 8-GPU ZeRO-2 CPU-offload training"
run_in_env bash scripts/train_zero2_offload.sh 8 "${TRAIN_OVERRIDES[@]}"
