#!/usr/bin/env bash
#SBATCH --job-name=isruc-pac-readout
#SBATCH --account=torch_pr_63_general
#SBATCH --partition=l40s_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

RELEASE=${1:?frozen source directory}
PYTHON=${2:?existing environment python}
JOB=${3:?absolute PAC evaluation job JSON}
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$RELEASE"
SCRIPT=scripts/experiments/pac_real_eeg_probe.py
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PYTHON" -u "$SCRIPT" "$JOB" --targets --smoke
"$PYTHON" -u "$SCRIPT" "$JOB" --smoke
"$PYTHON" -u "$SCRIPT" "$JOB" --targets
exec "$PYTHON" -u "$SCRIPT" "$JOB"
