#!/bin/bash
#SBATCH --job-name=score_tfm_tusz
#SBATCH --partition=mi2101x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/amd_env.sh
export PACLOCK_PROC=/work1/chenyuyou/yifanwang/Zhizhe
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MIOPEN_USER_DB_PATH=/tmp/miopen_sc_${SLURM_JOB_ID} MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_sc_${SLURM_JOB_ID}
mkdir -p "$MIOPEN_USER_DB_PATH" results/provisional
"$PACLOCK_PYTHON" -u smoke/score_ckpt.py --config configs/tusz/tusz-biot_tfm_host_s0.yaml \
  --ckpt runs/tusz-biot_tfm_host_20260918/seed0/best.pt --out results/provisional/tusz_tfm_host_s0.json
