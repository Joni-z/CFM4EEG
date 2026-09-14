# F: 16-band tokenizer follow-up, 2026-09-15

Authorized by the user after the 8-band residual/routing pilots were stopped.
Current origin/main finding reviewed: `26e2423`, plus COLLAB-PROTOCOL,
CANDIDATE-BOARD, N-BANDS-FINDING, residual gate, candidate review, historical
FINDINGS, protocol/budget/scheduling audits and handoff/status documents.
New code is isolated in `CFM4EEG-nbands-20260915`, branch
`codex/nbands-candidates-20260915`; the legacy checkouts and runs are preserved.

## Evidence and limitations

The 16-band n1 CHB run already completed six epochs with ordinary patience:
validation PR .670557898, versus old duplex .678727686. Both took about 18.7
hours. This makes a blind 12-hour single-card cap an inadequate completion plan.
TUEV n5 completed 20 epochs with validation kappa .633040061 and selected
index 8. Its descriptive test kappa .7391 is a single seed, not a replicated win.
The clean 8/16-band pairs support changed learning dynamics; later selection
alone does not prove better representations or make first-epoch selections invalid.
Do not select or overturn this screen on test metrics.

The new N-BANDS finding supersedes the retracted cross-corpus peak-index
ranking. `s4` is NOT a matched n5 control because coupling_self also differs.
Neither n_bands alone nor the full regularization package is established as
universally superior. This is a recipe-conditioned information-preservation test.

## Three designs, four new runs, seed 0

All use the n5 recipe: D128, depth 6, 16 bands, rotation, patch 50,
weight decay .05, dropout .35 and the existing five-transform augmentation.
Dataset-specific loss, batch 32, validation cadence, optimizer and 20-epoch
cosine schedule remain unchanged.

| Design | Run | Field changed relative to n5 |
|---|---|---|
| Coupling | CHB-MIT | none; n5 breadth test, never completed before |
| Residual | CHB-MIT and TUEV | model_kwargs.local_residual = true |
| Duplex | CHB-MIT | model_kwargs.tokenizer_mode = duplex |

Residual preserves the 16-row grid and adds a zero-initialized waveform
projection. Duplex exposes waveform rows independently but doubles the grid
from 16 to 32 rows. This is a representation and cost contrast, not an
equal-token-count causal comparison. Existing TUEV n5 seed 0 is the control;
no control is retrained just to fill a node. Historical runtime/source
identity is not established; record new source/runtime and cohort metadata.
A CHB-passing duplex would need its own TUEV check before any promotion.

## Predeclared decision

Use validation only and preserve the historical selected-metric convention.
- CHB-MIT: a candidate needs best validation PR >= .658727686 (within .02
  of existing duplex seed 0 .678727686). Compare the three same-recipe arms
  at equal processed-example counts as well as their selected checkpoints.
- TUEV residual: best validation kappa >= .623040061 (within .01 of n5),
  and balanced accuracy at that checkpoint no more than .02 below n5's
  selected-validation balanced accuracy, to be taken from the saved source
  metadata, never inferred from test balanced accuracy.
- Inspect TUEV class recall; a kappa improvement that loses minority classes
  is not sufficient. These margins are operational pilot gates, not statistical
  noninferiority or SOTA claims.
- A pass admits a replication proposal, not automatic multi-seed or pretraining
  spending. A failure downgrades this configuration, not all mixing from one seed.
- A time-budget or operator-stopped run is censored and cannot pass as a
  completed run or conclusively fail the design. No mid-step patience rewrite.

## Compute admission

`sbatch --test-only` confirmed mi2104x permits 24 hours; the 12-hour limit
applies to mi2101x. External allocation meter: used 1838.73 / 2250 at this
check. Billing weights are not a calibrated conversion to remaining node hours.

First run a <=25-minute four-GPU smoke, one real batch per configuration,
including optimizer/backward/finite gradients and all augmentation branches.
Exclude the first two steps and separately warm stochastic branches. The
16-band residual must preserve the base's initial parameters/logits and have
nonzero projection gradients. Preserve separate receipts for all four GPUs.

After reviewing measured speed and memory, submit ONE packed mi2104x
allocation (128 CPUs, 4 trainers initially, <=24h). CHB trainer cap 22h;
TUEV cap 7h. Full 20-epoch and six-epoch CHB projections are both reported.
Six epochs is an observed reference, not a promise of convergence or a change
to epochs/patience. Never run model computations on a login node. Output names
are new and collision-checked. No optimizer resume is implied.

This replaces the stopped AMD jobs 418923/418927/418928/419084/419085 and
pending Torch 17777804/17777805; their handover controller was terminated.
Torch TUSZ transplant jobs 17365676/17365677 were already running and are retained.
