# Quad16 shared 145 seed expansion — 2026-09-15

User authorized Quad16 seed confirmations, prioritizing shared 145. A128 remains historical reference; no duplicate architecture/pretraining launch.

- TUEV and ADFD already have seeds 0/1/2: do not repeat.
- SleepEDF, TUAR, Siena: add seeds 1/2 with the exact breadth seed0 recipe, changing seed only.
- CHBMIT: previously authorized seeds 1/2; existing full-SHA transfer worker invokes the updated wrapper after verification.
- 145 GPU6 currently belongs to another user's vLLM. Do not terminate or use occupied GPUs, even with 0% utilization.

`scripts/run_shared145.py` waits for verified data, then requires no compute process, memory <100 MiB and utilization 0. After obtaining our per-GPU flock it rechecks after 3 seconds. Per-run locks prevent duplicates. These locks coordinate our tasks only; this shared host has no global reservation service. Each task runs structural and real-data smoke tests before training. Training and smoke commands have time limits. Final result is required for success.

CHB wrapper is a post-verification entry point for the existing transfer_chbmit_145.py worker, not a general data verification command. It records that upstream verification succeeded and delegates to dynamic GPU admission, ignoring the obsolete fixed GPU argument.

145 runtime: /data2/zz5070/CFM4EEG-quad16-breadth-20260915
State receipts: results/shared145-<dataset>-s<seed>.json
Logs: logs/shared145-<dataset>-s<seed>.out (CHB uses logs/145-chbmit-s<seed>.out)
Local transfer workers: /Users/mr.z/PACLock-monitor/quad16-breadth/transfer_seed_breadth_145.py and transfer_chbmit_145.py. Transfers compare full source/destination SHA256 of manifest and six arrays before releasing training.

Initial admission: SleepEDF s1 GPU0, s2 GPU1; both real-data smokes passed (~0.15s/step excluding warmup, 6.91 GiB peak). TUAR/Siena have four waiting workers and active transfer automation. CHB remains transferring and automatically launches two workers after verification. Waiting processes consume no GPU. Do not report pending data as running training.
