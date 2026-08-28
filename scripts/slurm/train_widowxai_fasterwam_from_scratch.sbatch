#!/usr/bin/env bash
#SBATCH --job-name=fasterwam-2wrist-train
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

export RUN_ID="${RUN_ID:-widowxai_2wrist_vertical_scratch_${SLURM_JOB_ID}}"
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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"

RUN_DIR="${FASTERWAM_OUTPUT_ROOT}/${TASK_CONFIG}/${RUN_ID}"
RESOLVED_CONFIG="${RUN_DIR}/resolved_config.yaml"

mkdir -p \
  "${RESULT_ROOT}/slurm" \
  "${RESULT_ROOT}/pretrained" \
  "${RESULT_ROOT}/tmp/${SLURM_JOB_ID}" \
  "${RUN_DIR}"

cd "${TRAIN_REPO}"
ulimit -n 131072

run_in_env() {
  bash "${FASTWAM_REPO}/scripts/pixi_run.sh" \
    env "PYTHONPATH=${TRAIN_REPO}/src" "$@"
}

echo "[train] start=$(date -Is) host=$(hostname) job=${SLURM_JOB_ID} run_id=${RUN_ID}"
echo "[train] code_commit=$(git rev-parse HEAD)"
echo "[train] train_data=${TRAIN_DATASET}"
echo "[train] eval_data=${EVAL_DATASET}"
echo "[train] visual_input=drop_top_view left_wrist_top right_wrist_bottom final_size=384x320"
echo "[train] model_root=${MODEL_ROOT}"
echo "[train] output=${RUN_DIR}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

test "$(git rev-parse HEAD)" = "109e45a659cb12c4757753237c773d90c751b596"
test -x "${PIXI_HOME}/bin/pixi"
test -d "${TRAIN_DATASET}"
test -d "${EVAL_DATASET}"
test -d "${TEXT_CACHE}"
test -s "${STATS_PATH}"
test -s "${MODEL_ROOT}/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
test -s "${MODEL_ROOT}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors"
test -s "${MODEL_ROOT}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors"
test -s "${MODEL_ROOT}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors"

echo "[train] prebuilding DeepSpeed CPUAdam"
run_in_env python -u scripts/prebuild_ds_ops.py

if [[ ! -s "${ACTION_DIT_PATH}" ]]; then
  ACTION_DIT_TMP="${ACTION_DIT_PATH}.tmp-${SLURM_JOB_ID}"
  if [[ -e "${ACTION_DIT_TMP}" ]]; then
    echo "[train] refusing to overwrite temporary file: ${ACTION_DIT_TMP}" >&2
    exit 1
  fi
  echo "[train] generating SparseActionDiT initialization: ${ACTION_DIT_PATH}"
  run_in_env python -u scripts/preprocess_sparse_action_dit_backbone.py \
    --model-config configs/model/fasterwam.yaml \
    --output "${ACTION_DIT_TMP}" \
    --device cuda \
    --dtype bfloat16 \
    --condition-layers 0,4,8,12,16,20,24,28 \
    --apply-alpha-scaling true \
    --redirect-common-files false
  mv "${ACTION_DIT_TMP}" "${ACTION_DIT_PATH}"
fi

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
  "resume=null"
  "seed=42"
  "wandb.enabled=true"
  "wandb.mode=online"
  "wandb.name=${RUN_ID}"
  "wandb.group=widowxai_2wrist_vertical"
)

echo "[train] validating environment, initialization payload, and resolved configuration"
run_in_env python -u - "${ACTION_DIT_PATH}" <<'PY'
import sys

import deepspeed
import torch

path = sys.argv[1]
assert torch.cuda.device_count() == 8, torch.cuda.device_count()
gpu_names = [torch.cuda.get_device_name(i) for i in range(8)]
assert all(("H100" in name or "H200" in name) for name in gpu_names), gpu_names
assert deepspeed.__version__ == "0.19.4", deepspeed.__version__
payload = torch.load(path, map_location="cpu", weights_only=True)
assert payload["meta"]["condition_layers"] == [0, 4, 8, 12, 16, 20, 24, 28]
assert payload["policy"]["alpha_scaling"] is True
assert payload["backbone_state_dict"]
print(
    "preflight ok: "
    f"torch={torch.__version__} deepspeed={deepspeed.__version__} "
    f"gpu={gpu_names[0]} sparse_tensors={len(payload['backbone_state_dict'])}"
)
PY

run_in_env python -u scripts/train.py \
  "${TRAIN_OVERRIDES[@]}" \
  --cfg job --resolve > "${RESOLVED_CONFIG}"

run_in_env python -u - "${RESOLVED_CONFIG}" <<'PY'
import sys
from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
assert cfg.batch_size == 32
assert cfg.gradient_accumulation_steps == 4
assert cfg.batch_size * 8 * cfg.gradient_accumulation_steps == 1024
assert cfg.max_steps == 1600
assert cfg.save_every == 160
assert cfg.eval_every == 0
assert cfg.keep_last_n_states == 1
assert cfg.resume is None
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
assert list(cfg.model.condition_layers) == [0, 4, 8, 12, 16, 20, 24, 28]
print(
    "resolved config ok: drop_top_view=true, 2-wrist vertical 384x320, "
    "global_batch=1024, max_steps=1600, keep_last_state=1"
)
PY

sha256sum "${ACTION_DIT_PATH}" "${STATS_PATH}"
echo "[train] launching 8-GPU ZeRO-2 CPU-offload training"
run_in_env bash scripts/train_zero2_offload.sh 8 "${TRAIN_OVERRIDES[@]}"
