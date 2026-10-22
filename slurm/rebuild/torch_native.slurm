#!/bin/bash
#SBATCH --job-name=baseline-native
#SBATCH --account=torch_pr_63_general
#SBATCH --partition=l40s_public,h200_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
RELEASE=${1:?immutable source release}
PYTHON=${2:?absolute path to the baseline environment python}
CONFIG=${3:?config relative to release}
MODE=${4:?smoke or train}
: "${CFM_REBUILD_ROOT:?external output root}"
: "${PACLOCK_PROC:?parent of processed/}"
export CFM_REBUILD_ROOT PACLOCK_PROC PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
cd "$RELEASE"
exec "$PYTHON" -u scripts/rebuild/launch.py "$CONFIG" 0 "$MODE"
