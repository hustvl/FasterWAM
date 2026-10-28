#!/usr/bin/env bash
#SBATCH --job-name=fasterwam-widow-gpu-smoke
#SBATCH --partition=128-partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/data/fasterwam/slurm/%x-%j.out
#SBATCH --error=/lustre/data/fasterwam/slurm/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/user128004/gitrepo/FasterWAM}"
FASTWAM_REPO="${FASTWAM_REPO:-/home/user128004/gitrepo/FastWAM}"
CLIENT_ROOT="${CLIENT_ROOT:-/home/user128004/gitrepo/sbr-wm-deploy/trossen_ai_aloha}"
MODEL_ROOT="${MODEL_ROOT:-/lustre/data/fastwam_nakamura/checkpoints}"
RUN_DIR="${RUN_DIR:-/lustre/data/fasterwam/runs/widowxai_fasterwam_2wrist_vertical_384_1e-4/widowxai_2wrist_vertical_scratch_4443674}"
CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/checkpoints/weights/step_000960.pt}"
NORM_STATS="${NORM_STATS:-${RUN_DIR}/dataset_stats.json}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
ACTION_INFER_MODE="${ACTION_INFER_MODE:-one_pass_future_cache}"
SEED="${SEED:-42}"
SERVER_PORT="${SERVER_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 20000))}"
RESULT_DIR="${RESULT_DIR:-/lustre/data/fasterwam/validations/widowxai_server_gpu/job_${SLURM_JOB_ID:-local}}"
SERVER_LOG="${RESULT_DIR}/server.log"
CLIENT_RESULT="${RESULT_DIR}/client_result.json"
CONFIG_PATH="${CONFIG:-${RUN_DIR}/config.yaml}"
SERVER_PID=""

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  echo "[gpu-smoke] exit_status=${status} end=$(date -Is)"
  exit "${status}"
}
trap cleanup EXIT INT TERM

for required_path in \
  "${REPO_ROOT}/scripts/serve_widowxai_policy.py" \
  "${FASTWAM_REPO}/scripts/pixi_run.sh" \
  "${CLIENT_ROOT}/scripts/validate_remote_policy.py" \
  "${CONFIG_PATH}" \
  "${CHECKPOINT}" \
  "${NORM_STATS}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "[gpu-smoke] missing required path: ${required_path}" >&2
    exit 2
  fi
done

mkdir -p "${RESULT_DIR}" /lustre/data/fasterwam/slurm
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_ROOT}"
export DIFFSYNTH_SKIP_DOWNLOAD=true

SERVER_ARGS=(
  scripts/serve_widowxai_policy.py
  --run-dir "${RUN_DIR}"
  --checkpoint "${CHECKPOINT}"
  --norm-stats "${NORM_STATS}"
  --model-base-path "${MODEL_ROOT}"
  --host 127.0.0.1
  --port "${SERVER_PORT}"
  --device cuda:0
  --num-inference-steps "${NUM_INFERENCE_STEPS}"
  --action-infer-mode "${ACTION_INFER_MODE}"
  --seed "${SEED}"
)
if [[ -n "${CONFIG:-}" ]]; then
  SERVER_ARGS+=(--config "${CONFIG_PATH}")
fi
if [[ -n "${TASK:-}" ]]; then
  SERVER_ARGS+=(--task "${TASK}")
fi

echo "[gpu-smoke] start=$(date -Is) job=${SLURM_JOB_ID:-local} node=$(hostname)"
echo "[gpu-smoke] checkpoint=${CHECKPOINT}"
echo "[gpu-smoke] config=${CONFIG_PATH}"
echo "[gpu-smoke] result_dir=${RESULT_DIR} port=${SERVER_PORT}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

bash "${FASTWAM_REPO}/scripts/pixi_run.sh" python "${SERVER_ARGS[@]}" \
  >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
echo "[gpu-smoke] server_pid=${SERVER_PID} log=${SERVER_LOG}"

ready=false
for _ in $(seq 1 900); do
  if rg -q "\[fasterwam-server\] listening" "${SERVER_LOG}" 2>/dev/null; then
    ready=true
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[gpu-smoke] server exited before readiness" >&2
    tail -200 "${SERVER_LOG}" >&2 || true
    wait "${SERVER_PID}"
    exit 1
  fi
  sleep 1
done
if [[ "${ready}" != true ]]; then
  echo "[gpu-smoke] server readiness timed out" >&2
  tail -200 "${SERVER_LOG}" >&2 || true
  exit 1
fi

echo "[gpu-smoke] server ready; launching real remote-policy client"
bash "${FASTWAM_REPO}/scripts/pixi_run.sh" python \
  "${CLIENT_ROOT}/scripts/validate_remote_policy.py" \
  --host 127.0.0.1 \
  --port "${SERVER_PORT}" \
  --image-height 480 \
  --image-width 640 \
  --dof 14 \
  --history-steps 2 \
  --pings 2 \
  --output-json "${CLIENT_RESULT}"

test -s "${CLIENT_RESULT}"
rg -q '"status": "ok"' "${CLIENT_RESULT}"
echo "[gpu-smoke] PASS client_result=${CLIENT_RESULT}"
echo "[gpu-smoke] server log tail:"
tail -120 "${SERVER_LOG}"
