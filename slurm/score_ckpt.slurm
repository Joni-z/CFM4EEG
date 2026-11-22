#!/bin/bash
#SBATCH --job-name=score_ckpt
#SBATCH --partition=mi2101x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/amd_env.sh
export PACLOCK_PROC=/work1/chenyuyou/yifanwang/Zhizhe
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MIOPEN_USER_DB_PATH=/tmp/miopen_score_${SLURM_JOB_ID} MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_score_${SLURM_JOB_ID}
mkdir -p "$MIOPEN_USER_DB_PATH" results/provisional
for pair in "measured /home1/yifanwang/ckpt-live-145/chbmit_measured_s2.pt configs/twoaxis/chbmit_measured_s2_eval.yaml" \
            "weight   /home1/yifanwang/ckpt-live-145/chbmit_weight_s2.pt   configs/twoaxis/chbmit_weight_s2.yaml"; do
  set -- $pair; arm=$1; ck=$2; cfg=$3
  [ -f "$ck" ] || { echo "SKIP $arm: no $ck"; continue; }
  echo "=== $arm ==="
  "$PACLOCK_PYTHON" -u smoke/score_ckpt.py --config "$cfg" --ckpt "$ck" --out results/provisional/chbmit_${arm}_s2.json
done
