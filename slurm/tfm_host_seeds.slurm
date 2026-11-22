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
# Usage: sbatch --job-name=TFMhost_<ds> slurm/tfm_host_seeds.slurm configs/tfmhost/<ds>_biot_tfm_host.yaml 0 1 2
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/amd_env.sh
export PACLOCK_PROC=/work1/chenyuyou/yifanwang/Zhizhe
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cfg=$1; shift
"$PACLOCK_PYTHON" -u smoke/smoke_tfm_host.py --config "$cfg"
for s in "$@"; do
  "$PACLOCK_PYTHON" -u -m paclock_bench.training.train --config "$cfg" --seed "$s"
done
