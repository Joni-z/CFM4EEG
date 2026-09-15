#!/bin/bash
set -euo pipefail
cfg=${1:?config}; seed=${2:?seed}
mkdir -p results
# Actual loader/batch, preserved tail weights and finite-gradient admission.
timeout --kill-after=60s 45m "$PACLOCK_PYTHON" -u smoke/verify_quad_hosts.py --config "$cfg"
exec "$PACLOCK_PYTHON" -u -m paclock_bench.training.train --config "$cfg" --seed "$seed"
