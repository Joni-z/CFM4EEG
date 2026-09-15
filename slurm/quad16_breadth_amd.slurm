#!/bin/bash
#SBATCH --partition=mi2101x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --time=10:00:00
#SBATCH --signal=B:TERM@180
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
CFG=${1:?configuration required}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p results logs
source slurm/amd_env.sh
export HIP_VISIBLE_DEVICES=0
export MIOPEN_USER_DB_PATH=/tmp/miopen_breadth_${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
export PACLOCK_PROC=/work1/chenyuyou/yifanwang/Zhizhe
export PACLOCK_DATA=/work1/chenyuyou/yifanwang/Zhizhe/raw
bash slurm/smoke_gpu.slurm smoke/verify_modulation_carrier.py
bash slurm/smoke_gpu.slurm smoke/smoke_amd_partition.py --config "$CFG" --output "results/breadth-smoke-${SLURM_JOB_ID}.json"
"$PACLOCK_PYTHON" -u -m paclock_bench.training.train --config "$CFG" --seed 0 &
child=$!
trap 'kill -TERM "$child" 2>/dev/null || true; wait "$child" || true' TERM INT
wait "$child"
