#!/bin/bash
#SBATCH --job-name=F_QRange_TUEV
#SBATCH --partition=mi2101x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=09:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/amd_env.sh
export PACLOCK_PROC=/work1/chenyuyou/yifanwang/Zhizhe
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
test ! -e STOP_ALL_RANGE
# Use the normal GPU smoke entry point inside this one-card allocation.
bash slurm/smoke_gpu.slurm smoke/verify_quad_range.py --config configs/quad_range/tuev.yaml --output "results/range-contract-tuev-${SLURM_JOB_ID}.json"
bash slurm/smoke_gpu.slurm smoke/smoke_amd_partition.py --config configs/quad_range/tuev.yaml --require-budget-fit --output "results/range-smoke-tuev-${SLURM_JOB_ID}.json"
test ! -e STOP_ALL_RANGE
exec bash slurm/configs_packed.slurm configs/quad_range/tuev.yaml:0
