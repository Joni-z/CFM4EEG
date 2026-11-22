#!/bin/bash
#SBATCH --job-name=paper_diag_twoaxis
#SBATCH --partition=mi2101x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/amd_env.sh
export PACLOCK_PROC=/work1/chenyuyou/yifanwang/Zhizhe
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MIOPEN_USER_DB_PATH=/tmp/cfm_paper_${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"
run=runs/tuev-range_twoaxis_measured_20260917/seed0
out=results/paper_diagnostics_20260919
"$PACLOCK_PYTHON" -u scripts/paper_analyze_20260919.py --run "$run" --data "$PACLOCK_PROC/processed/tuev" --out "${out}_smoke" --smoke
"$PACLOCK_PYTHON" -u scripts/paper_analyze_20260919.py --run "$run" --data "$PACLOCK_PROC/processed/tuev" --out "$out"
