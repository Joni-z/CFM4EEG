#!/bin/bash
set -euo pipefail
cd /data2/zz5070/CFM4EEG-quad16-breadth-20260915
SEED=${1:?seed}
GPU=${2:?gpu}
[[ "$SEED" == 1 || "$SEED" == 2 ]] || exit 2
[[ "$GPU" == 2 || "$GPU" == 3 ]] || exit 2
exec 9>results/chbmit-seed${SEED}.lock
flock -n 9 || exit 3
[[ ! -f runs/chbmit-nb16_quadrature_20260915/seed${SEED}/result.json ]] || exit 0
export CUDA_VISIBLE_DEVICES=$GPU
export CFM_STANDALONE_RUN_ID=145-quad16-chbmit-s${SEED}
export PACLOCK_PROC=/data2/zz5070/CFM4EEG-data
export PYTHONPATH=/data2/zz5070/CFM4EEG-runtime${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/data2/zz5070/miniconda3/envs/py312/bin/python
used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
(( used < 100 )) || exit 4
"$PY" -u smoke/verify_modulation_carrier.py
"$PY" -u smoke/smoke_amd_partition.py --config configs/quad16_seeds/chbmit_s${SEED}.yaml --output results/chbmit-seed${SEED}-smoke.json
exec timeout --signal=TERM --kill-after=300s 24h "$PY" -u -m paclock_bench.training.train --config configs/quad16_seeds/chbmit_s${SEED}.yaml --seed "$SEED"
