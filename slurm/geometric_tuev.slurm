#!/bin/bash
#SBATCH --partition=mi2101x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/amd_env.sh
export PACLOCK_PROC=/work1/chenyuyou/yifanwang/Zhizhe
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HIP_VISIBLE_DEVICES=0
export MIOPEN_USER_DB_PATH=/tmp/cfft_miopen_${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/cfft_miopen_${SLURM_JOB_ID}
"$PACLOCK_PYTHON" -c 'import json; assert json.load(open("results/geo-smoke.json"))["status"] == "passed"'
exec "$PACLOCK_PYTHON" -u -m paclock_bench.training.train --config configs/geometric/tuev_aligned_s0.yaml --seed 0
