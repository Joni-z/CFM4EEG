#!/bin/bash
set -euo pipefail
cd /data2/zz5070/CFM4EEG-quad16-breadth-20260915
DS=${1:?dataset}
GPU=${2:?GPU index}
[[ "$DS" == adfd || "$DS" == sleepedf ]] || exit 2
[[ "$GPU" == 0 || "$GPU" == 1 ]] || exit 2
mkdir -p logs results
exec 9>"results/145-${DS}.lock"
flock -n 9 || exit 3
[[ ! -f runs/${DS}-quad16_breadth_20260915/seed0/result.json ]] || exit 0
export CUDA_VISIBLE_DEVICES=$GPU
export CFM_STANDALONE_RUN_ID=145-quad16-${DS}-20260915
export PACLOCK_PROC=/data2/zz5070/CFM4EEG-data
export PYTHONPATH=/data2/zz5070/CFM4EEG-runtime${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/data2/zz5070/miniconda3/envs/py312/bin/python
# Stop if another process has claimed the chosen GPU; no eviction.
used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
(( used < 100 )) || { echo "GPU occupied: $GPU $used MiB"; exit 4; }
"$PY" -u smoke/verify_modulation_carrier.py
"$PY" -u smoke/smoke_amd_partition.py --config configs/quad16_breadth/${DS}_s0.yaml --output results/145-smoke-${DS}.json
exec timeout --signal=TERM --kill-after=300s 10h "$PY" -u -m paclock_bench.training.train --config configs/quad16_breadth/${DS}_s0.yaml --seed 0
