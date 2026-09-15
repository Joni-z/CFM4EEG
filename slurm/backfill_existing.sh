#!/bin/bash
set -euo pipefail
cd /work1/chenyuyou/yifanwang/Zhizhe/CFM4EEG-anchored-watch-20260915
source slurm/amd_env.sh
export HANDOVER_END_EPOCH=$(date -d '2026-09-15 12:01:19' +%s)
exec "$PACLOCK_PYTHON" -u scripts/monitor/backfill_amd.py
