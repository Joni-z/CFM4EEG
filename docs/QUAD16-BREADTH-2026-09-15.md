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
