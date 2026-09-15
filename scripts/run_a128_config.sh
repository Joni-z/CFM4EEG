#!/bin/bash
set -euo pipefail
cfg=${1:?config}; seed=${2:?seed}
mkdir -p results logs
case "$cfg" in
 *a128_hosts*) smoke=smoke/verify_a128_hosts.py ;;
 *) smoke=smoke/smoke_amd_partition.py ;;
esac
extra=()
if [[ "$smoke" == smoke/smoke_amd_partition.py ]]; then extra=(--output "results/a128-smoke-${SLURM_JOB_ID:-145}-$(basename "$cfg").json"); fi
timeout --kill-after=60s 40m "$PACLOCK_PYTHON" -u "$smoke" --config "$cfg" "${extra[@]}"
exec "$PACLOCK_PYTHON" -u -m paclock_bench.training.train --config "$cfg" --seed "$seed"
