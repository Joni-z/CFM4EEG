> 当前状态见 [2026-09-17 进度总表](PROGRESS-2026-09-17.md)。以下为历史方案和提交记录，不代表当前队列。

# Sean controls: launch and comparison plan

Authorized 2026-09-17: launch reviewer-requested controls and host seed completion.
Independent checkout; no changes to existing live Range/window experiments.

## Frozen scope (22 fresh runs)

1. Reference-model identity (`u_hat=1`, including band zero), TUEV and CHB-MIT,
   seed 0. Reuse corresponding measured Range seed 0.
2. TUEV sample-shuffled Z, seed 0: roll the entire Z tensor by one along the
   batch axis in both training and evaluation, preserving electrode, patch and
   frequency-pair axes. Batch size must exceed one. No label-based pairing.
   This is retraining, distinct from the existing fixed-checkpoint intervention.
3. BIOT and CBraMod, TUEV and IIIC: measured seeds 1 and 2 (8 runs), reusing
   completed Range replacement seed 0.
4. BIOT CHB-MIT: measured seeds 1 and 2 (2 runs), reusing completed seed 0.
5. BIOT identity frontend on TUEV, IIIC and CHB-MIT: seeds 0, 1 and 2 (9 runs).
   Compare against the same Range frontend/adapter/host with measured modulation.
   This is the matched no-modulation replacement; historical native baselines
   remain separate contextual comparisons. CBraMod identity is not in this scope.

All measured/identity pairs retain configs except name, seed, and modulation mode.
Identity has identical allocated state tensors and initialization, but phase
projection is inactive: do not claim identical active trainable capacity.
Validation selects checkpoints; report all fixed seeds, including reversals.
Retain best.pt, configs, progress, results, and smoke receipts for release.

The CHB-MIT host recipe preserves the completed Range BIOT seed-0 batch size 128
and native loader; it does not retroactively match historical native batch 512.
No additional confidence gate or PAC-as-feature arm is silently substituted.

## Allocations

| Allocation | Work |
|---|---|
| AMD 422556, four MI210 | TUEV identity, shuffled Z, CBraMod measured s1/s2 |
| AMD 422557, four MI210, refill queue | BIOT TUEV/IIIC: measured s1/s2 and identity s0/s1/s2 |
| AMD 422561 / 422562, single MI210 each | CBraMod IIIC measured s1 / s2 |
| 145, shared vacancy/UUID/flock admission | CHB-MIT identity s0; BIOT five runs queued |

145 launcher PIDs: 1504803 (reference identity), 1504804--1504806 (BIOT
identity s0--s2), 1504807--1504808 (BIOT measured s1--s2).
GPU 7 admitted the reference identity; GPU 5 and 6 were occupied by others.
Waiting wrappers consume no GPU and recheck vacancy before starting.

Original CBraMod IIIC allocations 422558/422559 stopped during smoke, before
training: checkpoint outputs differed by roughly 2.4e-7, with identical weights.
Reload checks now require exact state-tensor equality plus output rtol=1e-5,
atol=1e-6; exact identity/shuffle operator checks remain zero-tolerance.
Restarted as 422561/422562. This is a test-tolerance correction, not model tuning.

## Contracts

`smoke/verify_sean.py`: measured operator equals frozen pre-change implementation;
all variant initial state tensors match measured; identity equals a+iw exactly,
including root; cross-sample shuffle equals measured with batch-rolled Z;
finite backward, nonzero waveform gradient, strict checkpoint reload.
Then real-data model smoke before fresh training. Host smoke additionally checks
native-tail initialization, host grid, same-token tail equivalence and actual
batch gradients/timing. Warm-up steps excluded. Every seed has separate receipts.

Live status: `results/<run>/seedN/pipeline.json`; AMD queue receipts in logs;
training outputs `runs/<run>/seedN/`. The new runs have not produced evidence yet.

## CHB-MIT diagnosis at launch

At evaluation 122 (epoch index 2, step 4400), best validation PR:
old Quad s0 0.4418576; old Quad s1 0.4441626; old Quad s2 0.4550369;
Range s0 0.3790443. Range is lagging, not numerically failed. Old Quad s0 later
reached validation PR 0.5873474 and test PR 0.7319168 at four epochs.
This trajectory does not guarantee Range recovery. Continue its existing stopping
rule; do not cancel based on a mismatched validation/test comparison.
