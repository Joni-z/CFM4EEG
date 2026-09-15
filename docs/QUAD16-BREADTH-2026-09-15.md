# Quad16 single-seed breadth screen (2026-09-15)

User authorized one candidate on Torch and Bridges-2. Candidate: Quad16, quadrature modulation, 16 bands, D128, depth6, n5 regularization. Existing seed0 test: TUEV kappa 0.7278934; CHBMIT PR-AUC 0.7319168. These are selection evidence, not independent confirmation. CHB validation 0.5873474 missed the earlier A128-based gate; this breadth screen is an explicitly exploratory bet, not a declaration that that gate passed.

## Fixed runs and budget
All seed0, configs/quad16_breadth/{dataset}_s0.yaml. No architecture or recipe tuning per dataset. Preserve dataset-specific loss, sample rate and class count from A128.
- b2: TUSZ training cap12h, Slurm14h; SleepEDF training cap8h, Slurm10h. L40S one GPU each.
- Torch: TUAR and Siena training cap8h each, Slurm10h each, one GPU per job, H200/L40S partitions.
Maximum requested GPU-hours44; training caps36. Release on completion. Never submit duplicate replicas. Smoke contract and real-batch numerical/gradient/timing checks precede training in each allocation. Warm up all augmentation branches and omit first two timing steps. A wall-budget stop is censored evidence, not model failure.

## Before-result decision rule
Validation selects checkpoints. Compare same split and metric to A128 seed0: TUSZ PR 0.3460204, SleepEDF kappa 0.6363762, TUAR kappa 0.6172852, Siena PR 0.4486612. Exploratory tolerance is -0.03 absolute: thresholds 0.3160204, 0.6063762, 0.5872852, 0.4186612. Require at least 3/4 completed tasks within tolerance, and no completed task worse by >0.10, before recommending a small host-replacement/pretraining pilot. This is an engineering screen, not statistical equivalence or a SOTA claim. Do not kill on one early epoch or retune from test results. Inspect final held-out results once and report any validation/test divergence. Numerical failures stop immediately. Censored runs do not count as passing. Host replacement and pretraining are not submitted by this batch.

External baseline performance remains the final target: the paper reports SleepEDF ContraWR kappa0.6916, TUAR FFCL kappa0.7025, Siena REVE PR0.5181, TUSZ FFCL PR0.5449. These are TEST references, never validation thresholds; protocol comparability still needs checking.

## Submission receipt, 2026-09-15 06:50 UTC
Training code/config commit: 18148c6. Isolated worktrees on both clusters; shared dirty checkouts preserved.
- b2 46027491: TUSZ, L40S,14h allocation.
- b2 46027492: SleepEDF,L40S,10h allocation.
- Torch 17832550: TUAR,H200/L40S,10h allocation.
- Torch 17832551: Siena,H200/L40S,10h allocation.
All four PENDING/Priority at verification. Smoke has NOT run yet; no new training metrics. Formal squeue start time unknown. Test-only estimates were b2 Sept19 and Torch Sept28; these are not promised dates. Testing Torch H200 alone with4h did not improve estimate. Other generic GPU partitions rejected by cluster admission (a100 invalid for this account/job); retained accepted public partitions, with no duplicate jobs.

Cross-cluster audit: canonical JSON of protocol+splits matches AMD exactly for all four manifests (including subject lists, class counts, window shapes). Creation timestamps can differ; neither timestamps nor metadata equality alone establish bitwise signal-array equality.

Logs in each isolated CFM4EEG-quad16-breadth-20260915 worktree: logs/Q16_*-<jobid>.out. GPU smoke receipts results/breadth-smoke-<jobid>.json; final results runs/<dataset>-quad16_breadth_20260915/seed0/result.json. No large data transfers and no other users' jobs modified.

## Torch account correction
Legacy pr_63_tandon_priority rejected as invalid by Slurm admission. User association torch_pr_63_tandon_advanced with h200_tandon passed test submission. In-place account update failed. Replacement jobs submitted held, original pending jobs 17832550/17832551 cancelled, release also failed (Unspecified error); held replacements 17833489/17833492 cancelled. Final unheld replacements: TUAR 17833526, Siena 17833527. Queue confirms advanced account and h200_tandon; initial TUAR reason QOSGrpGRES, Siena None pending scheduler evaluation. Same configs/seed/budget, explicit account+partition CLI overrides; no duplicate running jobs. Advanced account does not guarantee immediate scheduling. Launcher defaults corrected accordingly.

## AMD migration authorized by user
Torch pending jobs17833526/17833527 cancelled. Identical seed0 configs migrated to AMD mi2101x single-card nodes: TUAR420346 k006-004-v3; Siena420347 k006-004-v4. Each10h allocation, training cap8h. Both RUNNING and modulation contract passed; real-data smoke in progress at first check. Existing four-card allocation419232 preserved. b2 jobs unchanged. Launcher slurm/quad16_breadth_amd.slurm, code62efb9d. Submission filter reports allocation used1845.49 of2250.

## 145 extension
User authorized selective dataset transfer and use of145 GPUs. Transfer ADFD (~1GB) then SleepEDF (~4.4GB), using local relay and full SHA256 verification before launch. Standalone GPU0/1 only, explicit runtime ID (not simulated Slurm), check vacancy and acquire per-dataset lock, same numerical and real-data smoke, train cap8h/external10h limit. SleepEDF replaces pending b2 job46027492 only after destination data readiness; cancel pending source before launch to avoid duplicate. ADFD adds independent breadth evidence (A128 seed0 validation balanced accuracy0.4830212; exploratory -0.03 tolerance0.4530212) and does not retroactively replace any of the four original decision gates. Original repositories and conda environments preserved; only conda tarballs/index/log caches cleaned.145 data2 user directory68GB before cleaning; ~4.6GB cache freed,82GB free afterward.

## First completed breadth results
All comparisons below seed0 versus A128 seed0, same processed split.
- TUAR Quad16: val kappa0.6103729, test0.6243238,20epochs complete,5203s. A128 val0.6172852/test0.6289274. Near parity; validation tolerance passed.
- Siena Quad16: val PR0.4344283, test0.3450361,patience5epochs,5599s. A128 val0.4486612/test0.1098043. Validation tolerance passed, large single-seed test gain; still below paper REVE reference0.5181 (protocol comparison not established).
- ADFD Quad16 on145: val balanced accuracy0.5648308,test0.4214106,30epochs complete,3554s. A128 val0.4830212/test0.5616955. Validation improved but held-out score fell0.140285; do not hide this divergence or declare breadth success from validation gates alone. Thirty epochs inherited from task-specific A128 config. This additional dataset was registered before results, not substituted for original gate tasks.
- SleepEDF transfer verified by full SHA256; b2 pending46027492 cancelled,145 GPU1 training launched (supervisor2153031), first4epochs bestval kappa0.6239 versus A128seed0 best0.6363762. Provisional, not completed gate pass.
- TUSZ b2 46027491 stillPENDING/Priority.
AMD single-card allocations420346/420347 completed and released. Original419232 remains with Anchor16 and Residual16 pairs: CHB second epoch bestPR0.3581/0.2034; TUEV bestval kappa0.6487/0.6525. No candidate replacement justified yet.

## User-authorized seed confirmations
Add seed1/2 for Quad16 TUEV and CHBMIT, exact seed0 result config retained except seed (verified dictionary equality). Seed0 resides in CFM4EEG-nbands-20260915; new AMD seeds in this worktree,145 seeds in same-named runtime snapshot. Do not aggregate by folder name without both locations.
TUEV: existing419232 steps60/61, HIP0/2, seed1/2 respectively, srun launch PIDs869361/869362; logs/Q16_tuev-s{1,2}-419232.out. Structure contracts passed, real-data smoke running at registration. Existing CHB trainers HIP1/3 retained. Roughly7h allocation left versus seed0~4.1h. Allocation expiry/early batch termination would censor confirmation; do not count incomplete results as failures. No new AMD allocation purchased.
CHBMIT: single-card AMD admission limits discovered: mi2101x12h,mi3501x4h; prior seed0~12.7h. Therefore transfer43GB to145 and launch seed1/2 on GPU2/3,22h training caps24h external cap. Transfer local supervisor64857, script /Users/mr.z/PACLock-monitor/quad16-breadth/transfer_chbmit_145.py. Full SHA256 check before per-seed smoke and launch; independent dataset locks and GPU vacancy checks, no use of occupied GPU6. Destination free76GB before transfer. Snapshoted launcher scripts/run_quad16_seed_145.sh. No new architectures or duplicate seed0 runs.
