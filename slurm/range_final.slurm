#!/bin/bash
#SBATCH --job-name=RangeFinal
#SBATCH --partition=mi2104x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@180
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PACLOCK_PROC=/work1/chenyuyou/yifanwang/Zhizhe
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec bash slurm/configs_packed.slurm "$@"
