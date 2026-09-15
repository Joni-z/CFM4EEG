#!/bin/bash
#SBATCH --account=torch_pr_63_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --signal=B:TERM@180
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
CFG=${1:?configuration required}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p results logs
source /scratch/zz5070/miniconda3/etc/profile.d/conda.sh
conda activate py312
export PACLOCK_PROC=/scratch/zz5070/data
export PACLOCK_DATA=/scratch/zz5070/data/raw
python -u smoke/verify_modulation_carrier.py
python -u smoke/smoke_amd_partition.py --config "$CFG" --output "results/breadth-smoke-${SLURM_JOB_ID}.json"
python -u -m paclock_bench.training.train --config "$CFG" --seed 0 &
child=$!
trap 'kill -TERM "$child" 2>/dev/null || true; wait "$child" || true' TERM INT
wait "$child"
