#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-fasterwam}"
WANDB_NAME="${WANDB_NAME:-robotwin_fasterwam_3cam_384_1e-4}"

bash scripts/train_zero1.sh "${NPROC_PER_NODE}" \
  task=robotwin_fasterwam_3cam_384_1e-4 \
  "wandb.enabled=${WANDB_ENABLED}" \
  "wandb.project=${WANDB_PROJECT}" \
  "wandb.name=${WANDB_NAME}" \
  "$@"
