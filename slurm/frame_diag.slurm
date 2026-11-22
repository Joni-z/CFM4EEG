#!/bin/bash
#SBATCH --job-name=frame_diag
#SBATCH --partition=mi2101x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/amd_env.sh
export PACLOCK_PROC=/work1/chenyuyou/yifanwang/Zhizhe
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MIOPEN_USER_DB_PATH=/tmp/miopen_diag_${SLURM_JOB_ID} MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_diag_${SLURM_JOB_ID}
mkdir -p "$MIOPEN_USER_DB_PATH" results/frame_diag
R=runs
for pair in "tuev tuev-range_twoaxis_measured_20260917/seed2 configs/twoaxis/tuev_measured_s2.yaml" \
            "chbmit /home1/yifanwang/ckpt-backup-145/chbmit-range_twoaxis_measured_20260917/seed0 configs/twoaxis/chbmit_measured_s0.yaml"; do
  set -- $pair; ds=$1; run=$2; cfg=$3
  case "$run" in /*) ck=$run/best.pt ;; *) ck=$R/$run/best.pt ;; esac
  [ -f "$ck" ] || { echo "SKIP $ds: no $ck"; continue; }
  echo "=== $ds ==="
  "$PACLOCK_PYTHON" -u smoke/frame_dispersion.py --config "$cfg" --ckpt "$ck" --batches 8 \
      --out results/frame_diag/$ds.json
done
