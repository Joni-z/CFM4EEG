# Residual-16 backfill screen, 2026-09-15

User goal: replace failed configurations promptly and reuse idle allocated GPUs.
This extends the previous two-run standby limit to a second bounded pair of
seed-0 pilots, at most four simultaneous trainers total. No new node allocation.

Axis F mixing. Test local_residual=True on the existing 16-band n5 recipe:
PAC rotation plus learned per-band waveform residual, same 16 rows and encoder.
Relative to n5 only local_residual changes. Compared with anchored this is a
separate structural candidate, not a single-field causal contrast.

Motivation: N-BANDS-FINDING section 6 identifies the 8-band residual screen as
potentially recipe-confounded; 16-band residual has no matching output in the
current main runs archive. The new carrier CHB pilot completed below its gate;
anchored is unproven. A bounded additive alternative addresses the same
within-band information loss without committing to another pretraining run.
TUEV historical n5 seed0 and CHB historical duplex three seeds are reused.
CHB has no matched n5 strong-recipe control, so comparisons establish only
performance feasibility, not isolated residual benefit.

Start CHB on logical HIP slot3 now, TUEV on the next verified free slot.
Each config must pass real-batch smoke with all augmentation warmups and two
initial timing steps excluded. No repeated retries or overwriting run outputs.
CHB cap <=10.5h and TUEV <=6.5h, tightened to allocation end minus15min;
late admission requires >=9h/5h remaining respectively. Budget termination is
censored. No spending on another node or pretending a truncated run is failure.

Validation rules unchanged: CHB no performance stop before two epochs; best
PR<.15 after two or <.30 after three plus preceding20eval plateau<.01 permits
cooperative STOP preserving best. Nonfinite permits stop. Completed CHB target
valPR>=.658727686; TUEV selected val kappa>=.623040061 and balanced_acc>=.5045.
TUEV runs to normal termination; no test-set-driven decisions or family rejection.
No promotion beyond seed0 on these pilots alone.

The backfill controller checks process liveness and GPU masks inside allocation
419232 every10s. It pauses only batch-shell50528 to retain the allocation while
its own children run; all trainers continue. On completion/error it resumes the
shell. A separate heartbeat rescue resumes the shell if the controller dies.
The existing anchored controller and its dispatcher lease remain independent.
