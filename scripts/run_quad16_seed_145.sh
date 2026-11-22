#!/bin/bash
set -euo pipefail
cd /data2/zz5070/CFM4EEG-quad16-breadth-20260915
# Entry point for transfer_chbmit_145.py, after full source/destination SHA256 verification.
printf '%s\n' 'Verified by transfer_chbmit_145.py before launch' > /data2/zz5070/CFM4EEG-data/processed/chbmit/.cfm-sha256-verified
exec /data2/zz5070/miniconda3/envs/py312/bin/python -u scripts/run_shared145.py chbmit "${1:?seed}"
