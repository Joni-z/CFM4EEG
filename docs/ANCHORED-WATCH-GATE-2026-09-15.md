# Anchored carrier standby and bounded watch (2026-09-15)

Authorized: continuously watch AMD 419232 and Torch 17365676/17365677;
stop clearly unproductive current pilots and automatically launch the next design.
Scope: at most two new seed-0 training runs, CHB-MIT and TUEV. No pretraining,
replications, repeated crash retries or cancellation of unrelated/shared jobs.

Next hypothesis: preserve the complex waveform carrier c=r+iw at initialization,
then learn a bounded PAC rotation: h=c*(1+g*u)/|1+g*u|, g=.5*tanh(q), q=0.
There are 16 per-band gates, 16 token rows and the same 16-band carrier recipe.
The denominator is >=.5 for finite |u|<=1. Fixed-u norm preservation does not
imply invertibility of the whole tokenizer. Initially PAC gradients downstream
of the gate are zero, but gate gradients are nonzero; after opening the gate,
phase and waveform paths must both receive gradients. This can learn to ignore
PAC: a success will still require an ungated/raw-content control for attribution.
Existing carrier/quadrature runs are references, never overwritten or resumed.

Historical equal-progress audit, validation PR, batch32/eval200:
Duplex seed0: step7400=.1235, end epoch0=.2419, end epoch1=.4808;
seed1 .0929/.1709/.4312; seed2 .0291/.0483/.4631.
16-band n1 seed0 .0739/.1139/.4493. Do not compare early PR against final .6787.

Automatic stage-1 stop:
- no performance-based stop in the first two completed CHB epochs;
- at >=2 completed epochs: best PR <.15 AND improvement over the preceding
  20 evaluations <.01; at >=3 epochs use PR <.30 with the same plateau rule;
- nonfinite training loss or validation numbers: cooperative stop, preserve best;
- clear process failure (nonzero packed-run exit): record technical failure;
- stale logs/progress or SSH failure alone: alert, NEVER kill or submit;
- TUEV: wait for normal completion; require selected val kappa >=.623040061
  AND selected balanced accuracy >=.5045. Minority-class zeros are an alert,
  not a new, retroactively imposed kill rule;
- completed CHB: best PR >=.658727686; operator/budget stops are explicitly
  censored, not conclusive architecture failures.
No test-set numbers are used for switching. No single pilot rejects a family.

A qualifying failure opens ONE paired anchored pilot (CHB + TUEV), only after
GPU contract plus both actual-config AMD smoke receipts pass. Each Torch job
also runs its own real-config GPU smoke before training. No more than four
candidate trainers may run/be queued across original and fallback screens.
Use single GPU, h200_public or l40s_public, account torch_pr_63_general,
probe both partitions separately at trigger time and select the earlier admitted
start estimate (multi-partition requests gave misleadingly late starts).
Pending jobs have a 70h submission deadline, so they cannot start weeks later.
no GPU model constraint, 12h CHB/8h TUEV allocations, 10.5h CHB/6.5h TUEV
training caps in job-specific runtime configs. Scheduler probes showed a
much shorter L40S estimate for 12h than 24h. Predictions are not guarantees.
The optimizer/schedule remain unchanged; budget-ended pilots remain censored.
Do not start if measured training-only reference duration exceeds 75% of cap;
smoke failure cancels that admission with no resubmission loop.

Torch legacy TUSZ jobs remain observation-only: AUROC selects checkpoints;
their log PR is not the selection criterion. They retain their 23h cap.

Controller runs every 180s via local launchd, survives CLI/terminal exit;
requires this Mac awake and SSH reachable. Deadline 72h, state/log/decision
receipts on disk. No automatic external messages. STOP sentinel disables it.
