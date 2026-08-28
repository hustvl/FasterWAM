#!/usr/bin/env bash
#SBATCH --job-name=fasterwam-widow-dummy
#SBATCH --partition=128-partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/data/fasterwam/slurm/%x-%j.out
#SBATCH --error=/lustre/data/fasterwam/slurm/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/user128004/gitrepo/FasterWAM}"
FASTERWAM_PYTHON="${FASTERWAM_PYTHON:-/lustre/data/fastwam_nakamura/pixi/environments/fastwam-7884459458926539502/envs/default/bin/python}"

cd "${REPO_ROOT}"
mkdir -p /lustre/data/fasterwam/slurm
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

echo "[dummy-job] start=$(date -Is) job=${SLURM_JOB_ID:-local} node=$(hostname)"
echo "[dummy-job] test=binary_websocket_messagepack jpeg_decode ping reset select_action"
echo "[dummy-job] python=${FASTERWAM_PYTHON}"
test -x "${FASTERWAM_PYTHON}"
"${FASTERWAM_PYTHON}" scripts/test_widowxai_server_dummy.py
echo "[dummy-job] completed=$(date -Is)"
