#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${1:?Usage: bash scripts/train_zero2_offload.sh <nproc_per_node> [hydra_overrides...]}"
shift

EXTRA_ARGS=("$@")
NUM_MACHINES="${NNODES:-1}"
MACHINE_RANK="${NODE_RANK:-0}"
MAIN_PROCESS_IP="${MASTER_ADDR:-127.0.0.1}"
MAIN_PROCESS_PORT="${MASTER_PORT:-29500}"

is_integer() {
  [[ "${1}" =~ ^[0-9]+$ ]]
}

if ! is_integer "${NUM_MACHINES}" || ! is_integer "${MACHINE_RANK}"; then
  echo "Error: NUM_MACHINES (${NUM_MACHINES}) and MACHINE_RANK (${MACHINE_RANK}) must be integers." >&2
  exit 1
fi

extract_task_basename() {
  local cfg="$1"
  if [[ "${cfg}" == task/* ]]; then
    local name="${cfg#task/}"
    name="${name%.yaml}"
    echo "${name}"
    return 0
  fi
  return 1
}

TASK_BASENAME="train"
for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
  arg="${EXTRA_ARGS[$i]}"
  case "${arg}" in
    --config-name)
      if ((i + 1 < ${#EXTRA_ARGS[@]})); then
        next="${EXTRA_ARGS[$((i + 1))]}"
        if parsed="$(extract_task_basename "${next}")"; then
          TASK_BASENAME="${parsed}"
        fi
      fi
      ;;
    --config-name=*)
      cfg="${arg#--config-name=}"
      if parsed="$(extract_task_basename "${cfg}")"; then
        TASK_BASENAME="${parsed}"
      fi
      ;;
    task=*)
      cfg="${arg#task=}"
      TASK_BASENAME="${cfg%.yaml}"
      ;;
  esac
done

if [[ -z "${RUN_ID:-}" ]]; then
  if ((NUM_MACHINES <= 1)); then
    RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
  else
    echo "RUN_ID must be set for multi-node training." >&2
    exit 1
  fi
fi

echo "[launch] nproc_per_node=${NPROC_PER_NODE} num_machines=${NUM_MACHINES} machine_rank=${MACHINE_RANK} run_id=${RUN_ID}"

OUTPUT_ROOT="${FASTERWAM_OUTPUT_ROOT:-${FASTWAM_OUTPUT_ROOT:-./runs}}"
LAUNCH_ARGS=(
  --config_file scripts/accelerate_configs/accelerate_zero2_offload_ds.yaml
  --num_processes "$((NPROC_PER_NODE * NUM_MACHINES))"
  --num_machines "${NUM_MACHINES}"
  --machine_rank "${MACHINE_RANK}"
)
if ((NUM_MACHINES > 1)); then
  LAUNCH_ARGS+=(
    --main_process_ip "${MAIN_PROCESS_IP}"
    --main_process_port "${MAIN_PROCESS_PORT}"
  )
fi

accelerate launch \
  "${LAUNCH_ARGS[@]}" \
  scripts/train.py \
  "output_dir=${OUTPUT_ROOT}/${TASK_BASENAME}/${RUN_ID}" \
  "wandb.name=${TASK_BASENAME}" \
  "${EXTRA_ARGS[@]}"
