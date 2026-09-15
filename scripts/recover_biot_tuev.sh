#!/bin/bash
set -euo pipefail
cd /work1/chenyuyou/yifanwang/Zhizhe/CFM4EEG-transfer-pretrain-20260915
source slurm/amd_env.sh
export PACLOCK_DATA=/work1/chenyuyou/yifanwang/data
export PACLOCK_PROC=/work1/chenyuyou/yifanwang/Zhizhe
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$PACLOCK_PYTHON" -u -m preprocessing.biot_native --dataset tuev --out "$PACLOCK_PROC/processed_biot/tuev" --jobs 8
