#!/usr/bin/env bash
# Reuse the installed training Pixi environment without solving/updating it.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PIXI_ENV_DIR="${FASTERWAM_PIXI_ENV:-/lustre/data/fastwam_nakamura/pixi/environments/fastwam-7884459458926539502/envs/default}"
if [[ ! -x "${PIXI_ENV_DIR}/bin/python" ]]; then
  echo "Missing existing Pixi Python: ${PIXI_ENV_DIR}/bin/python; set FASTERWAM_PIXI_ENV" >&2
  exit 1
fi
export CONDA_PREFIX="${PIXI_ENV_DIR}"
export CUDA_HOME="${PIXI_ENV_DIR}"
export PATH="${PIXI_ENV_DIR}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TMPDIR="${TMPDIR:-/lustre/data/fasterwam/tmp}"
mkdir -p "${TMPDIR}"
exec "$@"
