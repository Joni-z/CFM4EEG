#!/bin/bash
#SBATCH --job-name=screen
#SBATCH --partition=mi2101x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
# Usage: sbatch --time=06:00:00 slurm/smoke_then_train.slurm config.yaml
# The YAML seed (default 0) is shared by preflight, smoke, and training.
set -euo pipefail
config=${1:?configuration required}
[[ "$SLURM_JOB_PARTITION" == mi2101x ]] || exit 2
cd "$SLURM_SUBMIT_DIR"
source slurm/amd_env.sh
export HIP_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MIOPEN_USER_DB_PATH="/tmp/miopen_screen_${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
seed=$("$PACLOCK_PYTHON" -c 'import sys,yaml; seed=yaml.safe_load(open(sys.argv[1])).get("seed",0); assert isinstance(seed,int) and not isinstance(seed,bool) and seed>=0, "seed must be a nonnegative integer"; print(seed)' "$config")
"$PACLOCK_PYTHON" scripts/slurm/run_packed.py --plan-only "$config:$seed"
"$PACLOCK_PYTHON" -u smoke/smoke_amd_partition.py --config "$config" --require-budget-fit
exec bash slurm/configs_packed.slurm "$config:$seed"
