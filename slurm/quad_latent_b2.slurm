#!/bin/bash
#SBATCH --partition=GPU-shared
#SBATCH --gpus=l40s-48:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --signal=B:TERM@180
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load pytorch/26.05-2.11-py3
export PYTHONPATH=/ocean/projects/cis260249p/qren2/Zhizhe/extra_packages:${PYTHONPATH:-}
export PACLOCK_PROC=/ocean/projects/cis260249p/qren2/Zhizhe
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cfg=configs/pretrain/quad16_latent_tueg.yaml
mkdir -p results logs
# Each invocation owns a separate smoke output; smoke never writes training weights.
timeout --kill-after=60s 30m python -u -m paclock_bench.training.pretrain_quad_latent --config "$cfg" --smoke
python -u -m paclock_bench.training.pretrain_quad_latent --config "$cfg" --resume &
child=$!
trap 'kill -TERM "$child" 2>/dev/null || true; wait "$child" || true' TERM INT
wait "$child"
